# filename: FAPM.py
# Repo-faithful (FAPM): near/far memory + fn_selector + sub-patch (coarse=4, fine=16) + scoring w*max(...)
# - NESSUNA normalizzazione ImageNet sugli input
# - Auto-build pickle se mancano, con chiavi compatibili con la repo ufficiale:
#     final_near_core_c, final_far_core_c, final_near_core_f, final_far_core_f, fn_selector_c, fn_selector_f
# - Integrato con le tue utility (datasets, loaders, valutazione)

import os, math, time, random, pickle
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# NEW: per valutazione image-level
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ==== tue utility (stesso naming che usi altrove) ====
from data_loader import build_ad_datasets, make_loaders, save_split_pickle, load_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- CONFIG ----------------
METHOD               = "FAPM"     # nome metodo per salvataggio pickle (cartelle tue utility)
CODICE_PEZZO         = "PZ3"

TRAIN_POSITIONS      = ["pos1"]
VAL_GOOD_PER_POS     = 20
VAL_GOOD_SCOPE       = ["pos1"]
VAL_FAULT_SCOPE      = ["pos1"]
GOOD_FRACTION        = 1.0  #0.2 #0.3 #0.5 #0.7  #1.0

PIECE_TO_POSITION = {
    "PZ1": "pos1",
    "PZ2": "pos5",
    "PZ3": "pos1",
    "PZ4": "pos1",
    "PZ5": "pos1",
}

IMG_SIZE             = 224
TEST_SEED  = 42  # controlla *solo* la scelta delle immagini di validation/test
TRAIN_SEED = 1  # lo puoi cambiare tu per variare il sottoinsieme di GOOD usati per il training

# Post-process
GAUSSIAN_SIGMA       = 4          # smoothing post-heatmap

# --- Coreset & Adaptive near/far (repo-style) ---
CORESET_RATIO        = 0.10       # near ratio
ADAPTIVE_RATIO       = 2.0        # far = near * ADAPTIVE_RATIO
SELECTOR_PERCENTILE  = 90         # celle "difficili" = d_imax in top p%

# Inference
K_NN                 = 4          # n_neighbors=4 (repo)
BATCH_TEST           = 1          # la repo testa con batch_size=1; qui supportiamo >1, ma RC omogenea

# ---------------------------------------------------


# ============ Backbone STPM (hook layer2 & layer3, AvgPool 3x3 s=1 p=1) ============
class STPM(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # carico i pesi ImageNet, ma NON applico normalizzazione ai dati (richiesto dall'utente)
        self.resnet = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
        self._feats: List[torch.Tensor] = []

        def hook_t(_m, _in, out):
            self._feats.append(out)

        # hook su layer2 e layer3 (come repo)
        self.resnet.layer2.register_forward_hook(hook_t)
        self.resnet.layer3.register_forward_hook(hook_t)
        self.device = device

    def _reset(self):
        self._feats = []

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._reset()
        _ = self.resnet(x)
        assert len(self._feats) == 2, "Hook non ha catturato layer2/layer3"
        # AvgPool 3x3 s=1 p=1 (repo)
        f = F.avg_pool2d(self._feats[0], kernel_size=3, stride=1, padding=1)  # layer2 ~ 28x28 per 224x224
        c = F.avg_pool2d(self._feats[1], kernel_size=3, stride=1, padding=1)  # layer3 ~ 14x14
        return f, c


# ==================== utilità geometriche & clustering ====================
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

def compute_cell_dimax(X: torch.Tensor, centers_idx: np.ndarray) -> float:
    """
    Dato X (N,d) e indici dei centroidi (m,), restituisce d_imax = max distanza punto->centro assegnato.
    """
    if X.numel() == 0 or centers_idx.size == 0:
        return 0.0
    C = X[centers_idx]                     # (m,d)
    D = pairwise_distances(X, C)           # (N,m)
    assign = torch.argmin(D, dim=1)        # (N,)
    radii = D[torch.arange(X.shape[0]), assign]
    return float(radii.max().item())


# ==================== feature → per cella e sub-patch ====================
def _split_into_cells_and_subpatches(feat: torch.Tensor, layer: str) -> torch.Tensor:
    """
    feat: (1, C, H, W). layer='fine'→ layer2 (28x28), 'coarse'→ layer3 (14x14)
    Restituisce:
      - fine:  (49, 16, C)  # 7x7 celle, ciascuna suddivisa in 4x4 = 16 sub-patch
      - coarse:(49,  4, C)  # 7x7 celle, ciascuna suddivisa in 2x2 = 4  sub-patch
    """
    assert feat.dim() == 4 and feat.shape[0] == 1
    _, C, H, W = feat.shape
    if layer == "fine":
        # 28x28 = (7*4) x (7*4)
        assert H % 7 == 0 and W % 7 == 0, "Dimensioni inattese per layer2"
        h_sub = H // 7  # 4
        w_sub = W // 7  # 4
    else:
        # 14x14 = (7*2) x (7*2)
        assert H % 7 == 0 and W % 7 == 0, "Dimensioni inattese per layer3"
        h_sub = H // 7  # 2
        w_sub = W // 7  # 2

    # [C, 7, h_sub, 7, w_sub] → [7,7,h_sub,w_sub,C] → [49, h_sub*w_sub, C]
    t = feat[0]  # (C,H,W)
    t = t.view(C, 7, h_sub, 7, w_sub).permute(1, 3, 2, 4, 0).contiguous()  # (7,7,h_sub,w_sub,C)
    t = t.view(7 * 7, h_sub * w_sub, C)  # (49, sub, C)
    return t


@torch.inference_mode()
def collect_train_vectors(
    device: torch.device,
    model: STPM,
    train_loader: torch.utils.data.DataLoader,
) -> Dict[str, List[List[np.ndarray]]]:
    """
    Raccoglie vettori per cella & subpatch da tutte le immagini GOOD di train.
    Ritorna dizionario:
      fine_list  : 49 liste, ognuna di 16 liste di np.ndarray (N_i x C)
      coarse_list: 49 liste, ognuna di  4 liste di np.ndarray (N_i x C)
    """
    fine_list  = [[[] for _ in range(16)] for __ in range(49)]
    coarse_list= [[[] for _ in range(4)]  for __ in range(49)]

    for (x, _y, _m) in tqdm(train_loader, desc="[FAPM|train] collect GOOD vectors (per cell & subpatch)"):
        x = x.to(device, non_blocking=True)
        f, c = model(x)  # (B,Cf,28,28), (B,Cc,14,14) aggregati (AvgPool 3x3)

        B = f.shape[0]
        for b in range(B):
            fb = f[b:b+1]
            cb = c[b:b+1]
            # split
            fine_ = _split_into_cells_and_subpatches(fb, layer="fine")    # (49,16,Cf)
            coarse_ = _split_into_cells_and_subpatches(cb, layer="coarse") # (49, 4,Cc)

            fine_np = fine_.cpu().numpy()
            coarse_np = coarse_.cpu().numpy()

            for cell in range(49):
                for sp in range(16):
                    fine_list[cell][sp].append(fine_np[cell, sp])
                for sp in range(4):
                    coarse_list[cell][sp].append(coarse_np[cell, sp])

        del x, f, c
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # stack in array (N x C) per ciascun (cell,sub)
    for cell in range(49):
        for sp in range(16):
            fine_list[cell][sp] = np.stack(fine_list[cell][sp], axis=0).astype(np.float32)
        for sp in range(4):
            coarse_list[cell][sp] = np.stack(coarse_list[cell][sp], axis=0).astype(np.float32)

    return {"fine_list": fine_list, "coarse_list": coarse_list}


# ==================== costruzione near/far + selector (repo-style) ====================
def _build_layer_memory(
    layer_list: List[List[np.ndarray]],          # 49 x S list of np arrays (N x C)
    coreset_ratio: float,
    adaptive_ratio: float,
    selector_percentile: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    layer_list: 49 celle × S subpatch; ognuno è (N x C)
    Ritorna tuple:
      near_bank: (49, S, M_near, C)
      far_bank : (49, S, M_far , C)
        selector : (49,) boolean (True => usa far a test-time)
    """
    S = len(layer_list[0])  # 16 (fine) o 4 (coarse)
    N = layer_list[0][0].shape[0]  # #GOOD train (costante)
    C = layer_list[0][0].shape[1]

    m_near = max(1, int(math.floor(coreset_ratio * N)))
    m_far  = max(1, int(math.floor(adaptive_ratio * m_near)))
    m_near = min(m_near, N)
    m_far  = min(m_far,  N)

    # Per coerenza tensori rettangolari, usiamo m_near/m_far costanti su tutte le celle
    near_bank = np.zeros((49, S, m_near, C), dtype=np.float32)
    far_bank  = np.zeros((49, S, m_far , C), dtype=np.float32)

    # 1) d_imax (base, con m_near) per ciascuna cella (media sui subpatch per robustezza)
    dmax_cells: List[float] = []
    per_cell_dmax: List[float] = [0.0 for _ in range(49)]

    for cell in range(49):
        d_acc = []
        for sp in range(S):
            X = torch.from_numpy(layer_list[cell][sp])
            idx = kcenter_greedy(X, m=m_near, seed=seed)
            dmax = compute_cell_dimax(X, idx)
            d_acc.append(dmax)
        per_cell_dmax[cell] = float(np.mean(d_acc))
        dmax_cells.append(per_cell_dmax[cell])

    thr = np.percentile(np.asarray(dmax_cells, dtype=np.float32), selector_percentile)
    selector = np.asarray([d >= thr for d in per_cell_dmax], dtype=np.bool_)

    # 2) costruzione near/far
    for cell in range(49):
        for sp in range(S):
            X = torch.from_numpy(layer_list[cell][sp])
            # near
            idx_n = kcenter_greedy(X, m=m_near, seed=seed)
            near_bank[cell, sp] = X[idx_n].cpu().numpy().astype(np.float32)
            # far
            idx_f = kcenter_greedy(X, m=m_far, seed=seed)
            far_bank[cell, sp]  = X[idx_f].cpu().numpy().astype(np.float32)

    return near_bank, far_bank, selector


@torch.inference_mode()
def build_memory_payload(
    device: torch.device,
    model: STPM,
    train_loader: torch.utils.data.DataLoader,
) -> Dict[str, np.ndarray]:
    # 1) raccogli i vettori per (cell, subpatch)
    raw = collect_train_vectors(device, model, train_loader)
    fine_list, coarse_list = raw["fine_list"], raw["coarse_list"]

    # 2) costruisci near/far + selector per ciascun layer
    near_f, far_f, sel_f = _build_layer_memory(
        fine_list, CORESET_RATIO, ADAPTIVE_RATIO, SELECTOR_PERCENTILE, TRAIN_SEED
    )
    near_c, far_c, sel_c = _build_layer_memory(
        coarse_list, CORESET_RATIO, ADAPTIVE_RATIO, SELECTOR_PERCENTILE, TRAIN_SEED
    )

    # 3) payload con chiavi repo
    payload = {
        "final_near_core_f": near_f,               # (49,16,M_near,Cf)
        "final_far_core_f":  far_f,                # (49,16,M_far ,Cf)
        "final_near_core_c": near_c,               # (49, 4,M_near,Cc)
        "final_far_core_c":  far_c,                # (49, 4,M_far ,Cc)
        "fn_selector_f":     sel_f.astype(np.uint8).reshape(-1, 1),  # (49,1) compatibile con squeeze()
        "fn_selector_c":     sel_c.astype(np.uint8).reshape(-1, 1),
        # info opzionali
        "cfg": {
            "coreset_ratio": float(CORESET_RATIO),
            "adaptive_ratio": float(ADAPTIVE_RATIO),
            "selector_percentile": float(SELECTOR_PERCENTILE),
            "k_nn": int(K_NN),
            "seed": int(TRAIN_SEED),
        }
    }
    return payload


# ==================== INFERENCE (repo-like) ====================
def _cdist_topk_mean(q: torch.Tensor, bank: torch.Tensor, k: int) -> torch.Tensor:
    """
    q:    (P, C)           # sub-patch query, P=16 (fine) o 4 (coarse)
    bank: (P, M, C)        # banca per cella, M = m_near o m_far
    return: (P, k)         # sempre k NN (con padding se M < k)
    """
    # Trattiamo P come batch: cdist((P,1,C),(P,M,C)) -> (P,1,M) -> (P,M)
    D = torch.cdist(q.unsqueeze(1), bank)   # (P, 1, M)
    D = D.squeeze(1)                        # (P, M)
    M = D.shape[1]

    # k effettivo = quanti vettori ci sono davvero in memory
    k_eff = min(k, M)
    vals, _ = torch.topk(D, k=k_eff, largest=False, dim=1)  # (P, k_eff)

    # Se ho meno di k vettori in memoria, replico l'ultimo vicino
    # per avere comunque (P, k) ed evitare mismatch di shape.
    if k_eff < k:
        last = vals[:, -1:].expand(-1, k - k_eff)           # (P, k - k_eff)
        vals = torch.cat([vals, last], dim=1)               # (P, k)

    return vals


@torch.inference_mode()
def inference_single_image(
    device: torch.device,
    model: STPM,
    x: torch.Tensor,                       # (1,3,H,W)
    mem: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Restituisce:
      anomaly_map_resized_blur (IMG_SIZE x IMG_SIZE, float32)
      image_score (float)
    """
    # 1) estrai feature
    f, c = model(x)  # (1,Cf,28,28) & (1,Cc,14,14)

    # 2) split in (49,sub,C)
    fine_q   = _split_into_cells_and_subpatches(f, layer="fine")    # (49,16,Cf)
    coarse_q = _split_into_cells_and_subpatches(c, layer="coarse")  # (49, 4,Cc)

    # 3) carica memory su device
    near_f = torch.from_numpy(mem["final_near_core_f"]).to(device)     # (49,16,Mn,Cf)
    far_f  = torch.from_numpy(mem["final_far_core_f"]).to(device)      # (49,16,Mf,Cf)
    near_c = torch.from_numpy(mem["final_near_core_c"]).to(device)     # (49, 4,Mn,Cc)
    far_c  = torch.from_numpy(mem["final_far_core_c"]).to(device)      # (49, 4,Mf,Cc)
    sel_f  = torch.from_numpy(mem["fn_selector_f"]).to(device).squeeze().bool()  # (49,)
    sel_c  = torch.from_numpy(mem["fn_selector_c"]).to(device).squeeze().bool()  # (49,)

    # 4) per cella: scegli near/far e calcola kNN
    score_patch_f = torch.zeros((49, 16, K_NN), device=device, dtype=torch.float32)
    score_patch_c = torch.zeros((49,  4, K_NN), device=device, dtype=torch.float32)

    for cell in range(49):
        # fine
        qf = fine_q[cell].to(device)  # (16,Cf)
        bf = (far_f[cell] if sel_f[cell] else near_f[cell])  # (16,M,Cf)
        score_patch_f[cell] = _cdist_topk_mean(qf, bf, K_NN)  # (16,k)

        # coarse
        qc = coarse_q[cell].to(device)  # (4,Cc)
        bc = (far_c[cell] if sel_c[cell] else near_c[cell])  # (4,M,Cc)
        score_patch_c[cell] = _cdist_topk_mean(qc, bc, K_NN)  # (4,k)

    # 5) allinea coarse (4) a fine (16) ripetendo ×4 (repo)
    score_patch_c_up = score_patch_c.repeat_interleave(4, dim=1)  # (49,16,k)

    # 6) somma fine+coarse → score_patch totale (49,16,k)
    score_patch = score_patch_f + score_patch_c_up

    # 7) anomaly map a 28x28 (repo: rearrange '(h1 h) (w1 w) v -> (h1 w1 h w) v' con h1=7,w1=4)
    #    Prima ricaviamo solo il primo NN per la mappa (indice 0)
    amap_49x16 = score_patch[:, :, 0]  # (49,16)
    # mapping a 28x28:
    #   input (49,16) = (h1*h, w1*w) con h1=7,h=7, w1=4,w=4  ->  (h1*w1, h*w) = (28,28)
    # costruiamo via view/permute per evitare dipendenze einops
    # Prima rimappiamo (49 -> 7x7) e (16 -> 4x4)
    amap = amap_49x16.view(7, 7, 4, 4).permute(0, 2, 1, 3).contiguous().view(28, 28)  # (28,28)

    # 8) image-level score (repo):
    #    score_patches: (49*16, k) = (28*28, k)
    score_patches = score_patch.view(49 * 16, K_NN)  # (784, k)
    # w = 1 - (max(softmax(N_b)) )
    Nb = score_patches[torch.argmax(score_patches[:, 0])]  # (k,)
    w = 1.0 - (torch.exp(Nb).max() / torch.exp(Nb).sum())
    image_score = float(w.item() * score_patches[:, 0].max().item())

    # 9) resize a IMG_SIZE e blur
    amap_res = F.interpolate(amap.unsqueeze(0).unsqueeze(0), size=IMG_SIZE,
                             mode="bilinear", align_corners=False).squeeze().cpu().numpy().astype(np.float32)
    if GAUSSIAN_SIGMA > 0 and HAS_SCIPY:
        amap_res = gaussian_filter(amap_res, sigma=GAUSSIAN_SIGMA)

    return amap_res, image_score


# ==================== VALIDAZIONE ====================
@torch.inference_mode()
def run_validation(
    device: torch.device,
    model: STPM,
    val_loader: torch.utils.data.DataLoader,
    mem_payload: Dict[str, np.ndarray],
) -> Dict:
    raw_score_maps: List[np.ndarray] = []
    img_scores: List[float] = []
    gt_img, gt_masks = [], []

    t0, N = time.time(), 0
    for (x, y, m) in tqdm(val_loader, desc="[FAPM|test] validation"):
        x = x.to(device, non_blocking=True)
        gt_img.extend(y.cpu().numpy())
        gt_masks.extend(m.cpu().numpy())

        B = x.shape[0]
        for b in range(B):
            amap, s = inference_single_image(device, model, x[b:b+1], mem_payload)
            raw_score_maps.append(amap)
            img_scores.append(s)
        N += B

        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fps = N / max(1e-9, (time.time() - t0))
    print(f"[FAPM] validation FPS ≈ {fps:.2f}")

    return {
        "raw_score_maps": raw_score_maps,
        "img_scores": np.asarray(img_scores, dtype=np.float32),
        "gt_img": np.asarray(gt_img, dtype=np.int32),
        "gt_masks": gt_masks,
        "fps": fps,
    }


# ==================== MAIN ====================
def main():
    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nessuna normalizzazione: transform=None
    transform = None

    # Datasets & loaders (GOOD/FAULT come da tue utility)
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=TEST_SEED,
        train_seed=TRAIN_SEED,
        transform=None,
        debug_print_val_paths=True,   # <<< accendi la stampa
    )
    TRAIN_TAG = meta["train_tag"]
    print("[meta]", meta)
    print(f"Train GOOD (pos {meta['train_positions']}): {meta['counts']['train_good']}")
    print(f"Val   GOOD: {meta['counts']['val_good']}")
    print(f"Val  FAULT (pos {meta['val_fault_positions']}): {meta['counts']['val_fault']}")
    print(f"Val  TOT: {meta['counts']['val_total']}")

    # === Unica modifica richiesta: batch di validazione = 1 (repo-faithful) ===
    train_loader, _ = make_loaders(train_set, val_set, batch_size=max(8, BATCH_TEST), device=device)
    _, val_loader   = make_loaders(train_set, val_set, batch_size=1,                   device=device)

    # Backbone
    model = STPM(device=device).to(device).eval()

    # Memory: carica pickle repo-style oppure costruisci
    need_build = False
    try:
        mem_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        required = {
            "final_near_core_c", "final_far_core_c",
            "final_near_core_f", "final_far_core_f",
            "fn_selector_c", "fn_selector_f"
        }
        if not required.issubset(set(mem_payload.keys())):
            print("[pickle] trovato ma con chiavi non compatibili → rebuild")
            need_build = True
    except FileNotFoundError:
        need_build = True

    if need_build:
        print("[pickle] Nessuna memory compatibile → build near/far + selector (repo-style)")
        mem_payload = build_memory_payload(device, model, train_loader)
        save_split_pickle(mem_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[pickle] Memory salvata (repo-style).")
    else:
        print("[pickle] Memory caricata da cache (repo-style).")

    # Validazione
    out = run_validation(device, model, val_loader, mem_payload)

    # ================= IMAGE-LEVEL  =================
    img_scores = out["img_scores"]                 # (N_img,)
    gt_img     = out["gt_img"].astype(np.int32)    # 0=good, 1=fault

    # ROC, AUC
    fpr, tpr, thresholds = roc_curve(gt_img, img_scores)
    auc_img = roc_auc_score(gt_img, img_scores)
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")

    # Soglia di Youden
    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])

    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_img, preds, labels=[0, 1]).ravel()
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    # Curva ROC 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={auc_img:.3f}")
    ax[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax[0].set_title("Image-level ROC")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend()
    plt.tight_layout()
    plt.show()

    print(f"[check] len(val_loader.dataset) = {len(val_loader.dataset)}")
    print(f"[check] len(img_scores)         = {len(img_scores)}")

    # ================ PIXEL-LEVEL  ================
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


def run_single_experiment():
    """
    Esegue un esperimento completo usando le variabili globali:
        CODICE_PEZZO
        GOOD_FRACTION
    Ritorna:
        (image_auroc, pixel_auroc, pixel_auprc, pixel_aucpro)
    """
    # seed come nel main
    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======= DATASET =======
    transform = None  # nessuna normalizzazione, come nel main

    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=TEST_SEED,
        train_seed=TRAIN_SEED,
        transform=None,
        debug_print_val_paths=True,   # <<< accendi la stampa
    )
    TRAIN_TAG = meta["train_tag"]
    # print("[meta]", meta)
    # print(f"Train GOOD (pos {meta['train_positions']}): {meta['counts']['train_good']}")
    # print(f"Val   GOOD: {meta['counts']['val_good']}")
    # print(f"Val  FAULT (pos {meta['val_fault_positions']}): {meta['counts']['val_fault']}")
    # print(f"Val  TOT: {meta['counts']['val_total']}")

    # loaders (stessa logica del main)
    train_loader, _ = make_loaders(train_set, val_set, batch_size=max(8, BATCH_TEST), device=device)
    _, val_loader   = make_loaders(train_set, val_set, batch_size=1,                   device=device)

    # ======= MODELLO =======
    model = STPM(device=device).to(device).eval()

    # ======= MEMORY (stessa logica del main, nessuna funzione nuova) =======
    need_build = False
    try:
        mem_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        required = {
            "final_near_core_c", "final_far_core_c",
            "final_near_core_f", "final_far_core_f",
            "fn_selector_c", "fn_selector_f"
        }
        if not required.issubset(set(mem_payload.keys())):
            print("[pickle] trovato ma con chiavi non compatibili → rebuild")
            need_build = True
    except FileNotFoundError:
        need_build = True

    if need_build:
        print("[pickle] Nessuna memory compatibile → build near/far + selector (repo-style)")
        mem_payload = build_memory_payload(device, model, train_loader)
        # save_split_pickle(mem_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[pickle] Memory salvata (repo-style).")
    else:
        print("[pickle] Memory caricata da cache (repo-style).")

    # ======= VALIDAZIONE =======
    out = run_validation(device, model, val_loader, mem_payload)

    img_scores = out["img_scores"]                 # (N_img,)
    gt_img     = out["gt_img"].astype(np.int32)    # 0=good, 1=fault

    # ======= IMAGE-LEVEL =======
    fpr, tpr, thresholds = roc_curve(gt_img, img_scores)
    auc_img = roc_auc_score(gt_img, img_scores)

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_img, preds, labels=[0, 1]).ravel()

    # ======= PIXEL-LEVEL (no visual per velocità) =======
    results = run_pixel_level_evaluation(
        score_map_list=list(out["raw_score_maps"]),
        val_set=val_set,
        img_scores=out["img_scores"],
        use_threshold="pro",
        fpr_limit=0.01,
        vis=False,
        vis_ds_or_loader=None
    )

    pixel_auroc   = float(results["curves"]["roc"]["auc"])
    pixel_auprc   = float(results["curves"]["pr"]["auprc"])
    pixel_auc_pro = float(results["curves"]["pro"]["auc"])
    
    
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}, good_frac={GOOD_FRACTION}): {auc_img:.3f}")
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG} | gf={GOOD_FRACTION}")

    return float(auc_img), pixel_auroc, pixel_auprc, pixel_auc_pro


def run_all_fractions_for_current_piece():
    """
    Esegue più esperimenti variando GOOD_FRACTION per il pezzo corrente (CODICE_PEZZO).
    Usa solo variabili GLOBALI.
    """
    global GOOD_FRACTION

    # stessa griglia che hai usato per i risultati SPADE
    good_fracs = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00,
    ]

    img_list   = []
    pxroc_list = []
    pxpr_list  = []
    pxpro_list = []

    for gf in good_fracs:
        GOOD_FRACTION = gf
        print(f"\n=== FAPM | PEZZO {CODICE_PEZZO}, FRAZIONE GOOD = {GOOD_FRACTION} ===")
        auc_img, px_auroc, px_auprc, px_aucpro = run_single_experiment()

        img_list.append(auc_img)
        pxroc_list.append(px_auroc)
        pxpr_list.append(px_auprc)
        pxpro_list.append(px_aucpro)

    print("\n### RISULTATI FAPM PER PEZZO", CODICE_PEZZO)
    print("good_fractions      =", good_fracs)
    print("image_level_AUROC   =", img_list)
    print("pixel_level_AUROC   =", pxroc_list)
    print("pixel_level_AUPRC   =", pxpr_list)
    print("pixel_level_AUC_PRO =", pxpro_list)

    return {
        "good_fractions": good_fracs,
        "image_auroc": img_list,
        "pixel_auroc": pxroc_list,
        "pixel_auprc": pxpr_list,
        "pixel_auc_pro": pxpro_list,
    }


def run_all_pieces_and_fractions():
    """
    Esegue TUTTI i pezzi e TUTTE le frazioni.
    Usa variabili GLOBALI sovrascritte ogni volta:
      - CODICE_PEZZO
      - TRAIN_POSITIONS, VAL_GOOD_SCOPE, VAL_FAULT_SCOPE
    """
    global CODICE_PEZZO, TRAIN_POSITIONS, VAL_GOOD_SCOPE, VAL_FAULT_SCOPE

    # scegli qui i pezzi che vuoi far girare
    # pieces = ["PZ1", "PZ2", "PZ3", "PZ4", "PZ5"]
    pieces = ["PZ4", "PZ5"]
    

    all_results = {}

    for pezzo in pieces:
        CODICE_PEZZO = pezzo

        if pezzo not in PIECE_TO_POSITION:
            raise ValueError(f"Nessuna posizione definita in PIECE_TO_POSITION per il pezzo {pezzo}")

        pos = PIECE_TO_POSITION[pezzo]

        TRAIN_POSITIONS = [pos]
        VAL_GOOD_SCOPE  = [pos]
        VAL_FAULT_SCOPE = [pos]

        print(f"\n\n============================")
        print(f"   FAPM - RUNNING PIECE: {CODICE_PEZZO}")
        print(f"   POSITION:            {pos}")
        print(f"============================")

        res = run_all_fractions_for_current_piece()
        all_results[pezzo] = res

    print("\n\n========================================")
    print("      FAPM - RIEPILOGO TOTALE")
    print("========================================\n")

    for pezzo, res in all_results.items():
        print(f"\n----- {pezzo} -----")
        print("good_fractions      =", res["good_fractions"])
        print("image_level_AUROC   =", res["image_auroc"])
        print("pixel_level_AUROC   =", res["pixel_auroc"])
        print("pixel_level_AUPRC   =", res["pixel_auprc"])
        print("pixel_level_AUC_PRO =", res["pixel_auc_pro"])

    return all_results


if __name__ == "__main__":
    # 1) SOLO 1 ESPERIMENTO (usa le globali in testa)
    # main()

    # 2) TUTTE LE FRAZIONI PER UN SOLO PEZZO (CODICE_PEZZO globale)
    # run_all_fractions_for_current_piece()

    # 3) TUTTI I PEZZI × TUTTE LE FRAZIONI
    run_all_pieces_and_fractions()
