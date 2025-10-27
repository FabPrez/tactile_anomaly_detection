# spade_tiled_runner.py
# SPADE "divide & conquer" sequenziale su N tile per immagine (solo full-res).
# Griglia a CONTEGGIO FISSO (es. 2x3 -> 6 tile).
# NON salva file per-tile; ricompone full-res e calcola anche image-level aggregato.

import os, gc
from contextlib import nullcontext
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import tqdm

# --- tuoi pacchetti ---
from data_loader import build_ad_datasets
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# ----------------- CONFIG -----------------
METHOD = "SPADE"
CODICE_PEZZO = "PZ3"

TRAIN_POSITIONS = ["pos2"]
VAL_GOOD_PER_POS = 20
VAL_GOOD_SCOPE = ["pos2"]
VAL_FAULT_SCOPE = ["pos2"]
GOOD_FRACTION = 0.3
SEED = 42

# Backbone input size
BACKBONE_IMG_SIZE = 224

# --- Griglia: CONTEGGIO FISSO ---
FIXED_ROWS = 2           # es.: 2 righe
FIXED_COLS = 2           # es.: 2 colonne  -> 4 tile
FIXED_OVERLAP = 0        # overlap opzionale (0, 16, 32, ...)

# Visual/valutazione
GAUSSIAN_SIGMA = 4
FPR_LIMIT = 0.01
SAVE_DIR = "./_tiled_outputs"
DO_RECOMPOSE = True
AUTO_SIZE_FROM_DATA = True  # usa HxW dal dataset

# Salvataggi
SAVE_FINAL_FULLRES = True   # salva SOLO pacchetto finale full-res (nessun file per-tile)

# Aggregazione degli image-level score dai tile: "mean" oppure "max"
IMG_SCORE_AGG = "mean"
# ------------------------------------------


# ---------- utils griglia (conteggio fisso) ----------
def compute_tile_grid_by_counts(H: int, W: int, n_rows: int, n_cols: int, overlap: int = 0):
    """
    Divide l'immagine in n_rows × n_cols rettangoli contigui (copertura completa),
    con possibilità di un piccolo overlap interno. Ritorna lista di tuple (y,x,h,w).
    """
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


# ---------- wrapper dataset per vista tile ----------
class TileViewDataset(Dataset):
    """
    Avvolge un dataset base (che restituisce TENSORI (C,H,W), label, mask).
    Ritaglia (y,x,h,w) sul full-res e riporta a 3x224x224 con normalizzazione ImageNet.
    Ritorna: (img_224, label, mask_tile_224_np, idx_orig)
    """
    def __init__(self, base_ds: Dataset, tile_rect: Tuple[int,int,int,int], out_size: int = 224):
        self.base = base_ds
        self.y, self.x, self.h, self.w = tile_rect
        self.out = out_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img_t, lbl, mask_t = self.base[i]     # img_t: (C,H,W) float[0,1]; mask_t: (H,W) uint8 0/1
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


# ---------- feature utils ----------
def l2norm(x: torch.Tensor, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def topk_cdist_streaming(X, Y, k=5, block_x=1024, block_y=4096, device=torch.device('cuda'), use_amp=True):
    X = X.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)
    is_cuda = (isinstance(device, torch.device) and device.type == 'cuda') or (str(device) == 'cuda')
    all_topv, all_topi = [], []
    for i in range(0, X.size(0), block_x):
        xi = X[i:i+block_x]
        vals_row, inds_row = None, None
        amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if (use_amp and is_cuda) else nullcontext()
        with amp_ctx:
            for j in range(0, Y.size(0), block_y):
                yj = Y[j:j+block_y]
                d = torch.cdist(xi, yj)
                v, idx = torch.topk(d, k=min(k, d.size(1)), dim=1, largest=False)
                idx = idx + j
                if vals_row is None:
                    vals_row, inds_row = v, idx
                else:
                    vals_row = torch.cat([vals_row, v], dim=1)
                    inds_row = torch.cat([inds_row, idx], dim=1)
                    vals_row, new_idx = torch.topk(vals_row, k=k, dim=1, largest=False)
                    inds_row = inds_row.gather(1, new_idx)
        all_topv.append(vals_row.float().cpu()); all_topi.append(inds_row.cpu())
    return torch.cat(all_topv, 0), torch.cat(all_topi, 0)


# ---------- core: SPADE per UN tile ----------
def run_spade_for_tile(train_loader: DataLoader,
                       val_loader: DataLoader,
                       device: torch.device,
                       top_k: int = 5,
                       gaussian_sigma: int = 4) -> Dict[str, object]:

    # Backbone
    weights = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model   = wide_resnet50_2(weights=weights).to(device)
    model.eval()

    # hook
    outputs = []
    def hook(module, inp, out): outputs.append(out)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    tr_feats = {'layer1': [], 'layer2': [], 'layer3': [], 'avgpool': []}
    te_feats = {'layer1': [], 'layer2': [], 'layer3': [], 'avgpool': []}
    gt_list, order_val_idx = [], []

    # ---- TRAIN feature ----
    for x, y, m, _ in tqdm(train_loader, desc='[tile] feature | train'):
        x = x.to(device, non_blocking=True)
        with torch.no_grad(): _ = model(x)
        for k, v in zip(tr_feats.keys(), outputs): tr_feats[k].append(v.detach().cpu())
        outputs.clear()
    for k in tr_feats: tr_feats[k] = torch.cat(tr_feats[k], 0)

    # ---- VAL feature ----
    for x, y, m, idx in tqdm(val_loader, desc='[tile] feature | val'):
        gt_list.extend(y.numpy().tolist() if isinstance(y, torch.Tensor) else list(y))
        order_val_idx.extend(idx.numpy().tolist() if isinstance(idx, torch.Tensor) else [int(idx)])
        x = x.to(device, non_blocking=True)
        with torch.no_grad(): _ = model(x)
        for k, v in zip(te_feats.keys(), outputs): te_feats[k].append(v.detach().cpu())
        outputs.clear()
    for k in te_feats: te_feats[k] = torch.cat(te_feats[k], 0)

    gt_np = np.asarray(gt_list, dtype=np.int32)

    # ----- Image-level KNN su avgpool (streaming)
    X = torch.flatten(te_feats['avgpool'], 1).to(torch.float32)
    Y = torch.flatten(tr_feats['avgpool'], 1).to(torch.float32)
    topk_values, topk_indexes = topk_cdist_streaming(
        X, Y, k=top_k, block_x=1024, block_y=4096, device=device, use_amp=(device.type=="cuda")
    )
    img_scores = topk_values.mean(dim=1).cpu().numpy()

    # ----- Pixel-level localization
    score_map_list: List[np.ndarray] = []
    for t_idx in tqdm(range(te_feats['avgpool'].shape[0]), '[tile] localization | val'):
        per_layer = []
        for layer_name in ['layer1', 'layer2', 'layer3']:
            test_feat = te_feats[layer_name][t_idx:t_idx+1].to(device)        # (1,C,H,W)
            test_feat = l2norm(test_feat, dim=1)

            gallery_feat = l2norm(tr_feats[layer_name].to(device), dim=1)      # (Ntr,C,H,W)

            Ntr, C, H, W = gallery_feat.shape
            gallery = gallery_feat.permute(0,2,3,1).reshape(Ntr*H*W, C).contiguous()
            query   = test_feat.permute(0,2,3,1).reshape(H*W, C).contiguous()

            B = 20000
            mins = []
            for s in range(0, gallery.shape[0], B):
                d = torch.cdist(gallery[s:s+B], query)       # (B, H*W)
                mins.append(d.min(dim=0).values)
            dist_min = torch.stack(mins, 0).min(0).values    # (H*W,)

            score_map = dist_min.view(1,1,H,W)
            score_map = F.interpolate(score_map, size=BACKBONE_IMG_SIZE, mode='bilinear', align_corners=False)
            per_layer.append(score_map.cpu())

        score_map = torch.mean(torch.cat(per_layer, 0), 0).squeeze().numpy().astype(np.float32)
        if gaussian_sigma > 0:
            score_map = gaussian_filter(score_map, sigma=gaussian_sigma)  # usa il parametro locale
        score_map_list.append(score_map)

    # free
    del tr_feats, te_feats, topk_values; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    return {
        "img_scores": img_scores,          # image-level per questo tile (ordine val_loader)
        "gt_np": gt_np,
        "topk_indexes": topk_indexes.numpy(),
        "score_map_list": score_map_list,  # heatmap 224x224 per questo tile
        "order_val_idx": order_val_idx,    # mapping indice locale -> indice globale validation
    }


# ---------- ricomposizione full-res ----------
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


# ---------- MAIN ----------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

    # Dataset base SENZA crop/resize, SOLO full-res
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
        rgb_policy="fullres_only",   # tiled => usa sempre le immagini full-res
    )
    print("[meta]", meta)

    # HxW dal dataset
    if AUTO_SIZE_FROM_DATA:
        sample_img, _, _ = train_set[0]
        _, H0, W0 = sample_img.shape
        TGT_H, TGT_W = int(H0), int(W0)
    else:
        raise RuntimeError("AUTO_SIZE_FROM_DATA=False non gestito qui")

    # Costruzione griglia
    grid = compute_tile_grid_by_counts(TGT_H, TGT_W, FIXED_ROWS, FIXED_COLS, overlap=FIXED_OVERLAP)
    N_tiles = len(grid)
    print(f"[grid] {N_tiles} tile -> {grid[:3]}...")

    tile_to_rect: Dict[int, Tuple[int,int,int,int]] = {i: r for i, r in enumerate(grid)}
    tile_val_heatmaps: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(N_tiles)}

    # Per aggregare gli image-level score dai tile
    tile_img_scores: Dict[int, Dict[int, float]] = {i: {} for i in range(N_tiles)}

    # ====== Loop sequenziale sui tile ======
    for t_id, rect in enumerate(grid):
        print(f"\n=== TILE {t_id+1}/{N_tiles} rect={rect} ===")

        train_tile_ds = TileViewDataset(train_set, rect, out_size=BACKBONE_IMG_SIZE)
        val_tile_ds   = TileViewDataset(val_set,   rect, out_size=BACKBONE_IMG_SIZE)

        pin = (device.type == "cuda")
        train_loader = DataLoader(train_tile_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_tile_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)

        out = run_spade_for_tile(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            top_k=5,
            gaussian_sigma=GAUSSIAN_SIGMA
        )

        order_val_idx: List[int]    = out["order_val_idx"]
        score_map_list: List[np.ndarray] = out["score_map_list"]
        img_scores_tile: np.ndarray = out["img_scores"]

        # heatmap per-tile in RAM (senza scrivere su disco)
        for local_i, base_idx in enumerate(order_val_idx):
            tile_val_heatmaps[t_id][base_idx] = score_map_list[local_i]
            tile_img_scores[t_id][base_idx]   = float(img_scores_tile[local_i])

        # cleanup
        del train_loader, val_loader, train_tile_ds, val_tile_ds, out, score_map_list, order_val_idx, img_scores_tile
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    # ====== RICOMPOSIZIONE & IMAGE-LEVEL AGGREGATO ======
    if DO_RECOMPOSE:
        print("\n[recompose] Ricompongo le heatmap full-res dal validation set...")
        num_val = len(val_set)

        # full-res heatmaps
        full_res_maps = recompose_full_heatmaps(
            num_val=num_val,
            grid=grid,
            tile_heatmaps=tile_val_heatmaps,
            tile_to_rect=tile_to_rect,
            out_H=TGT_H, out_W=TGT_W
        )

        # image-level aggregato dai tile (media o max)
        agg_scores = np.zeros(num_val, dtype=np.float32)
        for idx in range(num_val):
            vals = [tile_img_scores[t].get(idx) for t in range(N_tiles) if idx in tile_img_scores[t]]
            if len(vals) == 0:
                agg_scores[idx] = 0.0
            else:
                if IMG_SCORE_AGG == "max":
                    agg_scores[idx] = float(np.max(vals))
                else:
                    agg_scores[idx] = float(np.mean(vals))

        # valutazione stile SPADE (PRO pixel-level + image-level dai punteggi aggregati)
        results = run_pixel_level_evaluation(
            score_map_list=full_res_maps,
            val_set=val_set,
            img_scores=agg_scores,
            use_threshold="pro",
            fpr_limit=FPR_LIMIT,
            vis=True,
            vis_ds_or_loader=val_set
        )
        print_pixel_report(results, title=f"{METHOD}-TILED | {CODICE_PEZZO}  tiles={N_tiles}  agg={IMG_SCORE_AGG}")

        if SAVE_FINAL_FULLRES:
            np.savez_compressed(os.path.join(SAVE_DIR, f"{CODICE_PEZZO}_fullres_val_heatmaps.npz"),
                                **{str(i): m for i, m in enumerate(full_res_maps)})
            print("[save] full-res heatmaps salvate (pacchetto unico).")

    print("\n[done] Sequenza tiled completata.")
    

if __name__ == "__main__":
    main()
