# FAPM_paper_faithful_no_imagenet.py
# FAPM — versione "paper-faithful" aderente a 2211.07381v2 e allineata alle tue utility
# Modifiche implementate:
#  - L2-normalization delle feature (per-vettore) in build e in inferenza (niente ImageNet normalize sugli input)
#  - Trigger coreset basato su percentile dei raggi (es. 90°)
#  - RADDOPPIO DI K (solo per le celle triggerate) in inferenza
#  - Salvataggio delle mappe trigger per cella (trig_fine/trig_coarse) nel pickle
#  - K dei nearest neighbors configurabile (default 8)
#
# - Layer: WRN50-2 hook su layer2 (fine) e layer3 (coarse)
# - Patching: forzato a 7×7 per entrambi i layer (Np = 49)
# - Adaptive coreset (2-stadi): base ratio 10% → calcolo r_max e percentile → eventuale aumento m (opzionale)
# - Inference: K nearest neighbors (media dei K NN) per cella, con K raddoppiato sulle celle triggerate
# - Fusione layer: somma 1:1 (coarse↑ -> fine); image-score = max_fine + max_coarse
# - Post: Gaussian blur σ=4
# - Normalizzazione input: DISATTIVATA (nessuna ImageNet normalize) — si normalizzano le FEATURE (L2)

import os, math, time, random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision import transforms as T  # opzionale

# === tue utility ===
from data_loader import build_ad_datasets, make_loaders, save_split_pickle, load_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- CONFIG ----------------
METHOD               = "FAPM"
CODICE_PEZZO         = "PZ2"

TRAIN_POSITIONS      = ["pos5"]
VAL_GOOD_PER_POS     = 20
VAL_GOOD_SCOPE       = ["pos5"]
VAL_FAULT_SCOPE      = ["pos5"]
GOOD_FRACTION        = 1.0

IMG_SIZE             = 224
SEED                 = 42

# Post-process
GAUSSIAN_SIGMA       = 4            # smoothing post-UPS (paper)

# --- Adaptive coreset (paper) ---
CORESET_BASE_RATIO   = 0.10         # 10% base ratio
CORESET_MAX_FRAC     = 0.50         # bound superiore (safety)
CORESET_MIN          = 16           # minimo assoluto (se possibile)
UPSCALE_FACTOR       = 2.0          # aumento m opzionale (paper-like)
CORESET_PERCENTILE   = 90.0         # soglia statistica sui raggi (percentile)

# Normalizzazione input ImageNet esplicitamente DISATTIVATA
NORMALIZE_IMAGENET   = False

# Fusione layer (paper-like: somma 1:1)
WEIGHT_FINE          = 1.0
WEIGHT_COARSE        = 1.0

# Inference: numero di NN per patch (configurabile)
K_PATCH              = 8
# ---------------------------------------


# ===== helper L2-normalization delle feature =====
def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ============ Backbone STPM-like (hook layer2 & layer3) ============
class STPM(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # pesi ImageNet per l'encoder, ma NESSUNA normalizzazione sugli input (richiesta esplicita)
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        self.resnet = wide_resnet50_2(weights=weights).to(device).eval()
        self._feats: List[torch.Tensor] = []

        def hook_t(_m, _in, out):
            self._feats.append(out)

        # SOLO layer2 (fine) e layer3 (coarse)
        self.resnet.layer2[-1].register_forward_hook(hook_t)  # FINE
        self.resnet.layer3[-1].register_forward_hook(hook_t)  # COARSE
        self.device = device

    def _reset(self):
        self._feats = []

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._reset()
        _ = self.resnet(x)
        assert len(self._feats) == 2, "Hook non ha catturato layer2/layer3"
        # smoothing locale (AvgPool 3×3 s=1)
        f = F.avg_pool2d(self._feats[0], kernel_size=3, stride=1, padding=1)  # (B, Cf, Hf, Wf)
        c = F.avg_pool2d(self._feats[1], kernel_size=3, stride=1, padding=1)  # (B, Cc, Hc, Wc)
        # FORZA 7×7 per entrambi i layer (Np = 49)
        f = F.adaptive_avg_pool2d(f, (7, 7))
        c = F.adaptive_avg_pool2d(c, (7, 7))
        return f, c


# ============ Utility distanza & k-center ============
def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2.0)

def kcenter_greedy(X: torch.Tensor, m: int, seed: int = 0) -> np.ndarray:
    """
    Greedy k-center su righe di X (N,d). Ritorna indici numpy (m,).
    """
    N = X.shape[0]
    m = max(1, min(int(m), N))
    if N == 0:
        return np.zeros((0,), dtype=np.int64)
    rng = random.Random(seed)
    start = rng.randrange(N)
    centers = [start]
    min_d = pairwise_distances(X, X[start:start+1, :]).squeeze(1)  # (N,)
    for _ in range(1, m):
        idx = torch.argmax(min_d).item()
        centers.append(idx)
        d_new = pairwise_distances(X, X[idx:idx+1, :]).squeeze(1)
        min_d = torch.minimum(min_d, d_new)
    return np.array(centers, dtype=np.int64)

def _stack_or_empty(lst: List[np.ndarray]) -> np.ndarray:
    if len(lst) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(lst, axis=0).astype(np.float32)


# ============ Raccolta feature per cella (batch-safe) ============
@torch.inference_mode()
def collect_train_features_per_cell(
    device: torch.device,
    model: STPM,
    train_loader: torch.utils.data.DataLoader,
) -> Dict[str, object]:
    mem_fine = None
    mem_coarse = None

    for (x, _y, _m) in tqdm(train_loader, desc="[FAPM|train] collect GOOD features (per cell)"):
        x = x.to(device, non_blocking=True)
        f, c = model(x)  # (B,C,7,7) dopo pooling forzato

        if mem_fine is None:
            Hf, Wf = f.shape[2], f.shape[3]   # = 7
            Hc, Wc = c.shape[2], c.shape[3]   # = 7
            mem_fine   = [[[] for _ in range(Wf)] for __ in range(Hf)]
            mem_coarse = [[[] for _ in range(Wc)] for __ in range(Hc)]

        # Permute a (B,7,7,C) e L2-normalizza i VETTORI feature
        vf = f.permute(0, 2, 3, 1).contiguous()  # (B,7,7,Cf)
        vc = c.permute(0, 2, 3, 1).contiguous()  # (B,7,7,Cc)
        vf = l2n(vf, dim=-1)
        vc = l2n(vc, dim=-1)

        B = vf.shape[0]
        for b in range(B):
            for i in range(vf.shape[1]):
                for j in range(vf.shape[2]):
                    mem_fine[i][j].append(vf[b, i, j].detach().cpu().numpy())
            for i in range(vc.shape[1]):
                for j in range(vc.shape[2]):
                    mem_coarse[i][j].append(vc[b, i, j].detach().cpu().numpy())

        del x, f, c, vf, vc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert mem_fine is not None, "Nessuna feature raccolta: controlla che il train_loader abbia campioni GOOD."

    for i in range(len(mem_fine)):
        for j in range(len(mem_fine[0])):
            mem_fine[i][j] = _stack_or_empty(mem_fine[i][j])
    for i in range(len(mem_coarse)):
        for j in range(len(mem_coarse[0])):
            mem_coarse[i][j] = _stack_or_empty(mem_coarse[i][j])

    return {"fine": mem_fine, "coarse": mem_coarse}


# ============ Adaptive coreset 2-stadi (paper) ============

def _initial_m(Ng: int) -> int:
    m = max(CORESET_MIN, int(math.floor(Ng * CORESET_BASE_RATIO)))
    return min(m, int(math.floor(Ng * CORESET_MAX_FRAC)), Ng)

@torch.inference_mode()
def build_adaptive_coreset_paper(
    memory_raw: Dict[str, object],
    seed: int = SEED
) -> Dict[str, object]:
    """
    Per ogni cella (i,j) e per ciascun layer:
      - Applica coreset 2-stadi (base 10%, trigger su percentile dei raggi)
      - Salva una sola bank per cella
      - Salva anche il flag trigger per cella (per raddoppiare K in inferenza)
    """
    def process_layer(mem_layer: List[List[np.ndarray]]):
        H, W = len(mem_layer), len(mem_layer[0])
        keys = [[None for _ in range(W)] for __ in range(H)]
        trig = [[False for _ in range(W)] for __ in range(H)]
        for i in range(H):
            for j in range(W):
                X = mem_layer[i][j]
                if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
                    keys[i][j] = np.zeros((0, 1), dtype=np.float32)
                    trig[i][j] = False
                    continue

                Xt = torch.from_numpy(X.astype(np.float32))
                Xt = l2n(Xt, dim=-1)

                Ng   = Xt.shape[0]
                m1   = _initial_m(Ng)
                idx1 = kcenter_greedy(Xt, m=m1, seed=seed)
                C1   = Xt[idx1]

                # assegnazioni e raggio cluster (max distanza punto→centroide)
                D1      = pairwise_distances(Xt, C1)       # (N, m1)
                assign1 = torch.argmin(D1, dim=1)          # (N,)
                radii   = D1[torch.arange(Ng), assign1]    # (N,)
                r_np    = radii.detach().cpu().numpy()
                tau     = float(np.percentile(r_np, CORESET_PERCENTILE))
                r_max   = float(r_np.max())
                trigger = (r_max > tau)

                # Step-2 opzionale (aumento m) — mantiene coerenza col paper senza essere essenziale
                m_final = m1
                if trigger and m1 < min(int(math.floor(Ng * CORESET_MAX_FRAC)), Ng):
                    m2  = int(math.ceil(m1 * UPSCALE_FACTOR))
                    m2  = min(m2, int(math.floor(Ng * CORESET_MAX_FRAC)), Ng)
                    if m2 > m1:
                        idx2 = kcenter_greedy(Xt, m=m2, seed=seed+1)
                        C1   = Xt[idx2]
                        m_final = m2

                C = C1.detach().cpu().numpy().astype(np.float32)
                if C.shape[0] == 0:
                    mu = l2n(Xt.mean(dim=0, keepdim=True), dim=-1).cpu().numpy().astype(np.float32)
                    keys[i][j] = mu
                    trig[i][j] = False
                else:
                    keys[i][j] = C
                    trig[i][j] = bool(trigger)

        return keys, trig

    keys_f, trig_f = process_layer(memory_raw["fine"])
    keys_c, trig_c = process_layer(memory_raw["coarse"])

    return {
        "keys_fine":   keys_f,
        "keys_coarse": keys_c,
        "trig_fine":   trig_f,
        "trig_coarse": trig_c,
        "cfg": {
            "base_ratio": float(CORESET_BASE_RATIO),
            "max_frac": float(CORESET_MAX_FRAC),
            "min": int(CORESET_MIN),
            "percentile": float(CORESET_PERCENTILE),
            "upscale_factor": float(UPSCALE_FACTOR),
            "seed": int(SEED),
        }
    }


# ============ Inference (K NN per cella, con trigger che raddoppia K) ============
@torch.inference_mode()
def _cell_topk_mean_dist(q: torch.Tensor, bank_np: np.ndarray, device: torch.device, k: int) -> torch.Tensor:
    if not isinstance(bank_np, np.ndarray) or bank_np.ndim != 2 or bank_np.shape[0] == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    B = torch.from_numpy(bank_np).to(device=device, dtype=torch.float32)  # (M,C)
    # L2-normalizza query e banco (cautelativo, anche se già L2)
    q = l2n(q, dim=-1)
    B = l2n(B, dim=-1)
    d = torch.cdist(q.unsqueeze(0), B, p=2.0).squeeze(0)                  # (M,)
    k = min(k, d.numel())
    topk, _ = torch.topk(d, k=k, largest=False)
    return topk.mean()

@torch.inference_mode()
def distance_map_single_memory(
    feat_hw: torch.Tensor,                  # (1, C, 7, 7)
    keys_cells: List[List[np.ndarray]],
    trig_cells: List[List[bool]],
    device: torch.device,
    k_patch: int = K_PATCH
) -> torch.Tensor:
    """Score per cella = mean dei K NN verso la bank. Ritorna map (7,7)."""
    B, C, H, W = feat_hw.shape
    assert B == 1 and H == 7 and W == 7, "Le feature devono essere 7×7 (Np=49)."
    # L2-normalizza canale per canale prima del confronto
    V = feat_hw[0].permute(1, 2, 0).contiguous()  # (7,7,C)
    V = l2n(V, dim=-1)

    out = torch.zeros((H, W), dtype=torch.float32, device=device)
    for i in range(H):
        for j in range(W):
            q   = V[i, j]
            kij = k_patch * 2 if (trig_cells and trig_cells[i][j]) else k_patch
            dval = _cell_topk_mean_dist(q, keys_cells[i][j], device, kij)
            out[i, j] = dval
    return out


@torch.inference_mode()
def run_validation(
    device: torch.device,
    model: STPM,
    val_loader: torch.utils.data.DataLoader,
    mem_payload: Dict,
    img_size: int,
    gaussian_sigma: int
) -> Dict:
    k_f = mem_payload["keys_fine"]
    k_c = mem_payload["keys_coarse"]
    t_f = mem_payload.get("trig_fine", None)
    t_c = mem_payload.get("trig_coarse", None)

    # fallback per pickle legacy: niente trigger → K base
    if t_f is None: t_f = [[False]*7 for _ in range(7)]
    if t_c is None: t_c = [[False]*7 for _ in range(7)]

    raw_score_maps: List[np.ndarray] = []
    img_scores: List[float] = []
    gt_list, gt_mask_list = [], []

    t0 = time.time(); N = 0

    for (x, y, m) in tqdm(val_loader, desc="[FAPM|test] validation"):
        x = x.to(device, non_blocking=True)
        gt_list.extend(y.cpu().numpy())
        gt_mask_list.extend(m.cpu().numpy())

        f, c = model(x)  # (B,Cf,7,7), (B,Cc,7,7)

        # L2-normalizza le feature per robustezza (già normalizzate a build)
        f = l2n(f, dim=1)
        c = l2n(c, dim=1)

        B = f.shape[0]

        for b in range(B):
            fb = f[b:b+1]
            cb = c[b:b+1]

            # mappe distanza per layer (K NN mean) con K raddoppiato sulle celle triggerate
            map_f = distance_map_single_memory(fb, k_f, t_f, device=device, k_patch=K_PATCH)  # (7,7)
            map_c = distance_map_single_memory(cb, k_c, t_c, device=device, k_patch=K_PATCH)  # (7,7)

            # porta coarse a fine (qui già 7×7 → no resize; lasciamo per robustezza)
            map_c_up = F.interpolate(map_c.unsqueeze(0).unsqueeze(0), size=map_f.shape,
                                     mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

            # fusione (somma 1:1)
            amap_fine = WEIGHT_FINE * map_f + WEIGHT_COARSE * map_c_up  # (7,7)

            # image-level score = max(layer2) + max(layer3) (pesati)
            img_score = float(map_f.max().item()) * WEIGHT_FINE + float(map_c.max().item()) * WEIGHT_COARSE
            img_scores.append(img_score)

            # upsample a IMG_SIZE e smoothing
            amap_u = F.interpolate(amap_fine.unsqueeze(0).unsqueeze(0), size=img_size,
                                   mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            if gaussian_sigma > 0 and HAS_SCIPY:
                amap_u = gaussian_filter(amap_u, sigma=gaussian_sigma)
            raw_score_maps.append(amap_u.astype(np.float32))

        N += B
        del x, f, c
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fps = N / max(1e-9, (time.time() - t0))
    print(f"[FAPM] validation FPS ≈ {fps:.2f}")

    return {
        "raw_score_maps": raw_score_maps,
        "img_scores": np.asarray(img_scores, dtype=np.float32),
        "gt_img": np.asarray(gt_list, dtype=np.int32),
        "gt_masks": gt_mask_list,
        "fps": fps,
    }


# ============ Main ============
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nessuna normalizzazione/transform (coerente con richiesta)
    tfm = None
    if NORMALIZE_IMAGENET:
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    # Datasets & loaders
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=SEED,
        transform=tfm,  # None
    )
    TRAIN_TAG = meta["train_tag"]
    print("[meta]", meta)
    print(f"Train GOOD (pos {meta['train_positions']}): {meta['counts']['train_good']}")
    print(f"Val   GOOD: {meta['counts']['val_good']}")
    print(f"Val  FAULT (pos {meta['val_fault_positions']}): {meta['counts']['val_fault']}")
    print(f"Val  TOT: {meta['counts']['val_total']}")

    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=8, device=device)

    # Backbone
    model = STPM(device=device).to(device).eval()

    # Memory: carica oppure build
    try:
        mem_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        req = {"keys_fine", "keys_coarse", "trig_fine", "trig_coarse"}
        if not req.issubset(set(mem_payload.keys())):
            raise FileNotFoundError("pickle non conforme (serve single-memory per cella + trig map)")
        print(f"[pickle] Memory caricata da cache ({METHOD}).")
    except FileNotFoundError:
        print("[pickle] Nessuna memory -> raccolta GOOD + adaptive coreset 2-stadi (percentile) + trigger map")
        raw_mem = collect_train_features_per_cell(device=device, model=model, train_loader=train_loader)
        mem_payload = build_adaptive_coreset_paper(raw_mem, seed=SEED)
        save_split_pickle(mem_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[pickle] Memory salvata.")

    # Validazione
    out = run_validation(
        device=device,
        model=model,
        val_loader=val_loader,
        mem_payload=mem_payload,
        img_size=IMG_SIZE,
        gaussian_sigma=GAUSSIAN_SIGMA
    )

    # Valutazione pixel-level (tue utility)
    results = run_pixel_level_evaluation(
        score_map_list=list(out["raw_score_maps"]),
        val_set=val_set,
        img_scores=out["img_scores"],
        use_threshold="pro",
        fpr_limit=0.01,
        vis=True,
        vis_ds_or_loader=val_loader.dataset
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")


if __name__ == "__main__":
    main()
