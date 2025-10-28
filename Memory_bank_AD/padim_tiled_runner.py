# padim_tiled_runner.py
# PaDiM "divide & conquer" sequenziale su N tile per immagine (solo full-res).
# Griglia a CONTEGGIO FISSO (es. 2x3 -> 6 tile). Nessun center-crop.
# Per ogni tile: fit PaDiM (mean/cov) sul train, heatmap su validation,
# ricomposizione full-res e image-level score aggregato (mean/max) sui tile.

import os, gc
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# --- tuoi pacchetti (già esistenti nel tuo repo) ---
from data_loader import build_ad_datasets
from ad_analysis import run_pixel_level_evaluation, print_pixel_report


# ----------------- CONFIG -----------------
METHOD = "PaDiM-TILED"
CODICE_PEZZO = "PZ3"

TRAIN_POSITIONS = ["pos2"]
VAL_GOOD_PER_POS = 20
VAL_GOOD_SCOPE = ["pos2"]
VAL_FAULT_SCOPE = ["pos2"]
GOOD_FRACTION = 1.0
SEED = 42

# Backbone / PaDiM
BACKBONE_IMG_SIZE = 224
PADIM_D = 550          # canali selezionati (<= somma canali l1+l2+l3)
RIDGE = 0.01           # stabilizzazione cov per-pixel
GAUSSIAN_SIGMA = 4     # smooth finale delle mappe 224x224

# Griglia fissa (tiles)
FIXED_ROWS = 2
FIXED_COLS = 2
FIXED_OVERLAP = 0      # opzionale (0, 16, 32,...)

# Image-level score aggregation across tiles: "mean" o "max"
IMG_SCORE_AGG = "mean"

# Valutazione/vis
FPR_LIMIT = 0.01
SAVE_DIR = "./_padim_tiled_outputs"
DO_RECOMPOSE = True
AUTO_SIZE_FROM_DATA = True
SAVE_FINAL_FULLRES = True   # salva solo il pacchetto finale delle heatmap ricomposte
# ------------------------------------------


# -------- utils: griglia fissa --------
def compute_tile_grid_by_counts(H: int, W: int, n_rows: int, n_cols: int, overlap: int = 0):
    ys = np.linspace(0, H, n_rows + 1, dtype=int)
    xs = np.linspace(0, W, n_cols + 1, dtype=int)
    rects = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = ys[r], ys[r + 1]
            x0, x1 = xs[c], xs[c + 1]
            if overlap > 0:
                if r > 0:        y0 = max(0, y0 - overlap // 2)
                if r < n_rows-1: y1 = min(H, y1 + overlap // 2)
                if c > 0:        x0 = max(0, x0 - overlap // 2)
                if c < n_cols-1: x1 = min(W, x1 + overlap // 2)
            h = max(1, y1 - y0)
            w = max(1, x1 - x0)
            rects.append((int(y0), int(x0), int(h), int(w)))
    return rects


# -------- dataset vista-tile (su tensori CHW) --------
class TileViewDataset(Dataset):
    """
    Avvolge un dataset base (img_t float[0,1] shape (C,H,W), label, mask).
    Ritaglia (y,x,h,w) sul full-res e riporta a 3x224x224 con normalizzazione ImageNet.
    Ritorna: (img_224, label, mask_tile_224_np, idx_orig)
    """
    def __init__(self, base_ds: Dataset, tile_rect: Tuple[int,int,int,int], out_size: int = 224):
        self.base = base_ds
        self.y, self.x, self.h, self.w = tile_rect
        self.out = out_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        img_t, lbl, mask_t = self.base[i]     # img_t: (C,H,W)
        C, H, W = img_t.shape
        y, x, h, w = self.y, self.x, self.h, self.w
        y = max(0, min(y, H-1)); x = max(0, min(x, W-1))
        h = max(1, min(h, H - y)); w = max(1, min(w, W - x))

        img_tile = img_t[:, y:y+h, x:x+w]      # (C,h,w)
        img_tile = F.interpolate(img_tile.unsqueeze(0), size=(self.out, self.out),
                                 mode='bilinear', align_corners=False).squeeze(0)
        img_tile = (img_tile - self.mean) / self.std

        if mask_t is not None:
            m = mask_t[y:y+h, x:x+w].float().unsqueeze(0).unsqueeze(0)
            m = F.interpolate(m, size=(self.out, self.out), mode="nearest").squeeze().cpu().numpy().astype(np.float32)
        else:
            m = None
        return img_tile, int(lbl.item() if isinstance(lbl, torch.Tensor) else int(lbl)), m, i


# -------- PaDiM embedding utils --------
def embedding_concat_nn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor upsample + concat canali (come nel tuo PaDiM base)."""
    y_up = F.interpolate(y, size=(x.shape[-2], x.shape[-1]), mode='nearest')
    return torch.cat([x, y_up], dim=1)


class PaDiMBackbone:
    """Backbone WRN50-2 con hook a l1,l2,l3 per PaDiM."""
    def __init__(self, device):
        weights = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.model = wide_resnet50_2(weights=weights).to(device).eval()
        self.outputs = []
        def hook(_m, _in, out): self.outputs.append(out)
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.device = device

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,224,224) normalizzato ImageNet
        return: emb (B, Ctot, H, W)
        """
        self.outputs.clear()
        _ = self.model(x.to(self.device, non_blocking=True))
        l1, l2, l3 = [t.cpu() for t in self.outputs[:3]]
        emb = embedding_concat_nn(l1, l2)
        emb = embedding_concat_nn(emb, l3)
        return emb  # (B, Ctot, H, W)


# -------- PaDiM fit/predict per TILE --------
def padim_fit_tile(backbone: PaDiMBackbone,
                   loader: DataLoader,
                   padim_d: int,
                   ridge: float,
                   seed: int):
    """
    Due passaggi streaming su TILE: stima mean (pass1) poi cov (pass2).
    Ritorna dict con mean (d,L), cov (d,d,L), sel_idx (d,), shape(H,W).
    """
    rng = torch.Generator().manual_seed(seed)
    sel_idx = None
    d = None
    H = W = L = None

    # --- PASSO 1: mean ---
    N = 0
    sum_x = None  # (d,L)
    for (x, _, _, _) in tqdm(loader, desc="[tile|PaDiM] pass1 mean", leave=False):
        emb_b = backbone.extract(x)  # (B,Ctot,H,W)
        if sel_idx is None:
            C_total = emb_b.shape[1]
            d = min(padim_d, C_total)
            sel_idx = torch.randperm(C_total, generator=rng)[:d].tolist()

        emb_b = emb_b[:, sel_idx, :, :]  # (B,d,H,W)
        B = emb_b.shape[0]
        H, W = emb_b.shape[-2], emb_b.shape[-1]
        L = H * W
        E = emb_b.view(B, d, L).to(torch.float32)  # (B,d,L)

        if sum_x is None:
            sum_x = E.sum(dim=0)                   # (d,L)
        else:
            sum_x += E.sum(dim=0)
        N += B
        del emb_b, E, x
        if backbone.device.type == "cuda":
            torch.cuda.empty_cache()

    mean = (sum_x / float(max(N,1))).cpu().numpy().astype(np.float32)
    del sum_x

    # --- PASSO 2: cov ---
    cov = np.zeros((d, d, L), dtype=np.float32)
    TILE = 256
    for (x, _, _, _) in tqdm(loader, desc="[tile|PaDiM] pass2 cov", leave=False):
        emb_b = backbone.extract(x)                # (B,Ctot,H,W)
        emb_b = emb_b[:, sel_idx, :, :]            # (B,d,H,W)
        B = emb_b.shape[0]
        E = emb_b.view(B, d, L).numpy().astype(np.float32)  # (B,d,L)
        del emb_b
        for l0 in range(0, L, TILE):
            l1_ = min(l0 + TILE, L)
            diffs = E[:, :, l0:l1_] - mean[:, l0:l1_][None, :, :]  # (B,d,t)
            cov[:, :, l0:l1_] += np.einsum('bdt,bkt->dkt', diffs, diffs, optimize=True).astype(np.float32)
        del E, x
        if backbone.device.type == "cuda":
            torch.cuda.empty_cache()

    cov /= float(max(1, N - 1))
    cov += (ridge * np.eye(d, dtype=np.float32))[:, :, None]
    return {
        "mean": mean, "cov": cov,
        "sel_idx": np.asarray(sel_idx, dtype=np.int64),
        "shape": (int(H), int(W))
    }


def padim_predict_tile(backbone: PaDiMBackbone,
                       loader: DataLoader,
                       bank: Dict,
                       img_out_size: int,
                       gaussian_sigma: float):
    """
    Calcola heatmap 224x224 per ogni sample del validation loader del TILE.
    Ritorna: list(score_map_224), gt_labels(list[int]), order_val_idx(list[int]),
             img_scores_tile(np.ndarray, per-sample image-level del tile).
    L'image score per tile è il max della mappa normalizzata (per coerenza col PaDiM base).
    """
    mean    = bank["mean"]            # (d,L)
    cov     = bank["cov"]             # (d,d,L)
    sel_idx = bank["sel_idx"]         # (d,)
    H, W    = bank["shape"]
    L       = H * W
    d       = sel_idx.shape[0]

    score_maps = []
    gt_list = []
    order_idx = []

    # per normalizzare a [0,1] (solo per image-score)
    all_vals_min, all_vals_max = np.inf, -np.inf
    first_pass_raw = []

    # --- estrazione & distanze (pass unico) ---
    for (x, y, m, idx) in tqdm(loader, desc="[tile|PaDiM] predict", leave=False):
        gt_list.extend(y.numpy().tolist() if isinstance(y, torch.Tensor) else list(y))
        order_idx.extend(idx.numpy().tolist() if isinstance(idx, torch.Tensor) else [int(idx)])

        emb_b = backbone.extract(x)                 # (B,Ctot,H,W)
        emb_b = emb_b[:, sel_idx, :, :]            # (B,d,H,W)
        B = emb_b.shape[0]
        E = emb_b.view(B, d, L).numpy().astype(np.float32)  # (B,d,L)

        TILE = 256
        dist2_LB = np.empty((L, B), dtype=np.float32)
        for l0 in range(0, L, TILE):
            l1_ = min(l0 + TILE, L)
            t = l1_ - l0
            diffs = np.transpose(E[:, :, l0:l1_], (2, 0, 1)) - mean[:, l0:l1_].T[:, None, :]  # (t,B,d)
            cov_t = np.transpose(cov[:, :, l0:l1_], (2, 0, 1)).copy()                         # (t,d,d)
            eps = 1e-2
            cov_t += eps * np.eye(d, dtype=np.float32)[None, :, :]
            cov_t_t = torch.from_numpy(cov_t)              # (t,d,d)
            diffs_t = torch.from_numpy(diffs)              # (t,B,d)
            Lfac = torch.linalg.cholesky(cov_t_t)          # (t,d,d)
            diffsT = diffs_t.transpose(1, 2).contiguous()  # (t,d,B)
            sol = torch.cholesky_solve(diffsT, Lfac)       # (t,d,B)
            dist2_tB = (diffsT * sol).sum(dim=1)           # (t,B)
            dist2_LB[l0:l1_, :] = dist2_tB.cpu().numpy().astype(np.float32)

        dist_arr = np.sqrt(dist2_LB.T).astype(np.float32).reshape(B, H, W)  # (B,H,W)
        first_pass_raw.append(dist_arr)

        all_vals_min = min(all_vals_min, dist_arr.min())
        all_vals_max = max(all_vals_max, dist_arr.max())

        del emb_b, E, x
        if backbone.device.type == "cuda":
            torch.cuda.empty_cache()

    # --- upsample + gaussian + normalizzazione per image-score ---
    img_scores_tile = []
    for dist_arr in first_pass_raw:
        dist_t  = torch.from_numpy(dist_arr).unsqueeze(1)  # (B,1,H,W)
        score_b = F.interpolate(dist_t, size=img_out_size, mode='bilinear',
                                align_corners=False).squeeze(1).numpy()  # (B,224,224)
        for i in range(score_b.shape[0]):
            if gaussian_sigma > 0:
                score_b[i] = gaussian_filter(score_b[i], sigma=gaussian_sigma)

            # normalizzazione solo per image-score
            s = (score_b[i] - all_vals_min) / (all_vals_max - all_vals_min + 1e-12)
            img_scores_tile.append(float(s.max()))

            # mappa RAW (non normalizzata) per la ricomposizione pixel-level
            score_maps.append(score_b[i].astype(np.float32))

    return score_maps, gt_list, order_idx, np.asarray(img_scores_tile, dtype=np.float32)


# -------- ricomposizione full-res --------
def cosine_window(h, w):
    wy = 0.5*(1 - np.cos(2*np.pi*(np.arange(h)/(h-1)))) if h > 1 else np.ones(1)
    wx = 0.5*(1 - np.cos(2*np.pi*(np.arange(w)/(w-1)))) if w > 1 else np.ones(1)
    return np.outer(wy, wx).astype(np.float32)

def recompose_full_heatmaps(num_val: int,
                            grid: List[Tuple[int,int,int,int]],
                            tile_heatmaps: Dict[int, Dict[int, np.ndarray]],
                            tile_to_rect: Dict[int, Tuple[int,int,int,int]],
                            out_H: int, out_W: int) -> List[np.ndarray]:
    acc = [np.zeros((out_H, out_W), np.float32) for _ in range(num_val)]
    wgt = [np.zeros((out_H, out_W), np.float32) for _ in range(num_val)]
    for t_id, rect in tile_to_rect.items():
        y, x, h, w = rect
        win = cosine_window(h, w)
        for vidx, hmap224 in tile_heatmaps[t_id].items():
            tile_hw = F.interpolate(
                torch.from_numpy(hmap224)[None,None],
                size=(h, w), mode="bilinear", align_corners=False
            ).squeeze().numpy().astype(np.float32)
            acc[vidx][y:y+h, x:x+w] += tile_hw * win
            wgt[vidx][y:y+h, x:x+w] += win
    final = []
    for i in range(num_val):
        m = np.divide(acc[i], np.maximum(wgt[i], 1e-6)).astype(np.float32)
        final.append(m)
    return final


# ----------------- MAIN -----------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

    # Dataset base SENZA crop/resize, SOLO full-res (policy dedicata)
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=None,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=SEED,
        transform=None,
        rgb_policy="fullres_only",   # importantissimo per i tile
    )
    print("[meta]", meta)

    # Dimensioni reali
    if AUTO_SIZE_FROM_DATA:
        sample_img, _, _ = train_set[0]
        _, H0, W0 = sample_img.shape
        TGT_H, TGT_W = int(H0), int(W0)
    else:
        raise RuntimeError("AUTO_SIZE_FROM_DATA=False non gestito qui")

    # Griglia
    grid = compute_tile_grid_by_counts(TGT_H, TGT_W, FIXED_ROWS, FIXED_COLS, overlap=FIXED_OVERLAP)
    N_tiles = len(grid)
    print(f"[grid] {N_tiles} tile (es. primi 3): {grid[:3]}")

    tile_to_rect: Dict[int, Tuple[int,int,int,int]] = {i: r for i, r in enumerate(grid)}
    tile_val_heatmaps: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(N_tiles)}
    tile_img_scores: Dict[int, Dict[int, float]] = {i: {} for i in range(N_tiles)}

    # Backbone condiviso (riusato per tutti i tile)
    backbone = PaDiMBackbone(device)

    # ===== loop sui tile (sequenziale) =====
    for t_id, rect in enumerate(grid):
        print(f"\n=== TILE {t_id+1}/{N_tiles} rect={rect} ===")

        train_tile_ds = TileViewDataset(train_set, rect, out_size=BACKBONE_IMG_SIZE)
        val_tile_ds   = TileViewDataset(val_set,   rect, out_size=BACKBONE_IMG_SIZE)

        pin = (device.type == "cuda")
        train_loader = DataLoader(train_tile_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_tile_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)

        # Fit PaDiM (mean/cov) sul TRAIN del tile
        bank = padim_fit_tile(backbone, train_loader, PADIM_D, RIDGE, seed=1024)

        # Predict heatmaps su VALIDATION del tile (+ image-level per-tile)
        score_map_list, gt_list, order_val_idx, img_scores_tile = padim_predict_tile(
            backbone, val_loader, bank, img_out_size=BACKBONE_IMG_SIZE, gaussian_sigma=GAUSSIAN_SIGMA
        )

        # Accumula in RAM (no salvataggi per-tile)
        for local_i, base_idx in enumerate(order_val_idx):
            tile_val_heatmaps[t_id][base_idx] = score_map_list[local_i]
            tile_img_scores[t_id][base_idx]   = float(img_scores_tile[local_i])

        # cleanup per liberare memoria
        del train_loader, val_loader, train_tile_ds, val_tile_ds, score_map_list, order_val_idx, img_scores_tile, bank
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ===== ricomposizione + image-level aggregato =====
    if DO_RECOMPOSE:
        print("\n[recompose] Ricompongo le heatmap full-res dal validation set...")
        num_val = len(val_set)

        # full-res pixel-level
        full_res_maps = recompose_full_heatmaps(
            num_val=num_val,
            grid=grid,
            tile_heatmaps=tile_val_heatmaps,
            tile_to_rect=tile_to_rect,
            out_H=TGT_H, out_W=TGT_W
        )

        # image-level aggregato dai tile
        agg_scores = np.zeros(num_val, dtype=np.float32)
        for idx in range(num_val):
            vals = [tile_img_scores[t].get(idx) for t in range(N_tiles) if idx in tile_img_scores[t]]
            if len(vals) == 0:
                agg_scores[idx] = 0.0
            else:
                agg_scores[idx] = float(np.max(vals) if IMG_SCORE_AGG == "max" else np.mean(vals))

        # valutazione stile SPADE/PaDiM (PRO pixel-level + image-level aggregato)
        results = run_pixel_level_evaluation(
            score_map_list=full_res_maps,
            val_set=val_set,
            img_scores=agg_scores,
            use_threshold="pro",
            fpr_limit=FPR_LIMIT,
            vis=True,                      # visual “come in SPADE/PaDiM”
            vis_ds_or_loader=val_set
        )
        print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO} tiles={N_tiles} agg={IMG_SCORE_AGG}")

        if SAVE_FINAL_FULLRES:
            np.savez_compressed(os.path.join(SAVE_DIR, f"{CODICE_PEZZO}_padim_fullres_val_heatmaps.npz"),
                                **{str(i): m for i, m in enumerate(full_res_maps)})
            print("[save] full-res heatmaps salvate (pacchetto unico).")

    print("\n[done] PaDiM tiled completato.")


if __name__ == "__main__":
    main()
