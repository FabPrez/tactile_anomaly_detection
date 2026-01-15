# fapm_tiled_runner.py
# FAPM "divide & conquer" sequenziale su tile 224x224 (sliding grid, overlap opzionale).
# Ricomposizione full-res con cosine window + image-level aggregato (mean/max sui tile).

import os, gc, random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Backbone & FAPM core dal tuo file "FAPM.py"
from FAPM import (
    STPM,
    build_memory_payload,
    inference_single_image,
)

# --- tue utility comuni ---
from data_loader import build_ad_datasets
from ad_analysis import run_pixel_level_evaluation, print_pixel_report


# ----------------- CONFIG -----------------
METHOD = "FAPM_TILED"
CODICE_PEZZO = "PZ1"

TRAIN_POSITIONS = ["pos1"]
VAL_GOOD_PER_POS = 20
VAL_GOOD_SCOPE = ["pos1"]
VAL_FAULT_SCOPE = ["pos1"]
GOOD_FRACTION = 0.6

TEST_SEED  = 42  # controlla SOLO la scelta delle immagini di validation/test
TRAIN_SEED = 42  # controlla SOLO la scelta del sottoinsieme GOOD usato nel training

# Dimensione tile / input backbone STPM
BACKBONE_IMG_SIZE = 224

# --- Griglia: MISURA FISSA 224x224 (sliding grid) ---
FIXED_OVERLAP = 0        # overlap opzionale (0, 16, 32, ...)

# Visual/valutazione
FPR_LIMIT = 0.01
SAVE_DIR = "./_fapm_tiled_outputs"
DO_RECOMPOSE = True
AUTO_SIZE_FROM_DATA = True  # usa HxW dal dataset full-res

# Salvataggi
SAVE_FINAL_FULLRES = True   # salva SOLO pacchetto finale full-res (nessun file per-tile)

# Aggregazione degli image-level score dai tile: "mean" oppure "max"
IMG_SCORE_AGG = "max"
# ------------------------------------------


# ---------- utils griglia (MISURA FISSA 224x224) ----------
def compute_tile_grid_by_size(H: int, W: int, tile_h: int = 224, tile_w: int = 224,
                              overlap: int = 0) -> List[Tuple[int,int,int,int]]:
    """
    Griglia sliding a misura fissa:
    - tile_h/tile_w ~ 224
    - overlap in pixel (0,16,32,...)
    Copre tutta l'immagine (gli ultimi step sono clampati ai bordi).
    Ritorna lista di (y,x,h,w).
    """
    assert tile_h > 0 and tile_w > 0
    stride_h = max(1, tile_h - overlap)
    stride_w = max(1, tile_w - overlap)

    rects: List[Tuple[int,int,int,int]] = []
    y = 0
    while True:
        x = 0
        h = min(tile_h, H - y)
        while True:
            w = min(tile_w, W - x)
            rects.append((int(y), int(x), int(h), int(w)))
            if x + tile_w >= W:
                break
            x += stride_w
            if x + tile_w > W:
                x = max(0, W - tile_w)
        if y + tile_h >= H:
            break
        y += stride_h
        if y + tile_h > H:
            y = max(0, H - tile_h)
    return rects


# ---------- wrapper dataset per vista tile ----------
class TileViewDatasetTrain(Dataset):
    """
    Avvolge un dataset base (che restituisce TENSORI (C,H,W), label, mask).
    Ritaglia (y,x,h,w) sul full-res e riporta a 3x224x224.
    Nessuna normalizzazione ImageNet (coerente con FAPM_repo_faithful_no_imagenet).
    Ritorna: (img_224, label, mask_tile_224_np)
    """
    def __init__(self, base_ds: Dataset, tile_rect: Tuple[int,int,int,int], out_size: int = 224):
        self.base = base_ds
        self.y, self.x, self.h, self.w = tile_rect
        self.out = out_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img_t, lbl, mask_t = self.base[i]     # img_t: (C,H,W) float[0,1]; mask_t: (H,W) uint8 0/1
        C, H, W = img_t.shape
        y, x, h, w = self.y, self.x, self.h, self.w
        y = max(0, min(y, H-1)); x = max(0, min(x, W-1))
        h = max(1, min(h, H - y)); w = max(1, min(w, W - x))

        img_tile = img_t[:, y:y+h, x:x+w]      # (C,h,w)
        img_tile = F.interpolate(
            img_tile.unsqueeze(0), size=(self.out, self.out),
            mode='bilinear', align_corners=False
        ).squeeze(0)                           # (C, out, out)

        if mask_t is not None:
            m = mask_t[y:y+h, x:x+w].float().unsqueeze(0).unsqueeze(0)
            m = F.interpolate(m, size=(self.out, self.out), mode="nearest").squeeze().cpu().numpy().astype(np.float32)
        else:
            m = None

        return img_tile, int(lbl), m


class TileViewDatasetVal(TileViewDatasetTrain):
    """
    Come TileViewDatasetTrain ma ritorna anche l'indice originale dell'immagine.
    Ritorna: (img_224, label, mask_tile_224_np, idx_orig)
    """
    def __getitem__(self, i):
        img_tile, lbl, m = super().__getitem__(i)
        return img_tile, lbl, m, i


# ---------- ricomposizione full-res ----------
def cosine_window(h, w):
    wy = 0.5*(1 - np.cos(2*np.pi*(np.arange(h)/(h-1)))) if h > 1 else np.ones(1, dtype=np.float32)
    wx = 0.5*(1 - np.cos(2*np.pi*(np.arange(w)/(w-1)))) if w > 1 else np.ones(1, dtype=np.float32)
    return np.outer(wy, wx).astype(np.float32)


def recompose_full_heatmaps(num_val: int,
                            tile_heatmaps: Dict[int, Dict[int, np.ndarray]],
                            tile_to_rect: Dict[int, Tuple[int,int,int,int]],
                            out_H: int, out_W: int) -> List[np.ndarray]:
    """
    Ricompone le heatmap 224x224 dei vari tile in heatmap full-res HxW,
    usando una cosine window per pesare le zone di overlap.
    """
    acc = [np.zeros((out_H, out_W), np.float32) for _ in range(num_val)]
    wgt = [np.zeros((out_H, out_W), np.float32) for _ in range(num_val)]

    for t_id, rect in tile_to_rect.items():
        y, x, h, w = rect
        win = cosine_window(h, w)
        for vidx, hmap224 in tile_heatmaps[t_id].items():
            tile_hw = F.interpolate(
                torch.from_numpy(hmap224)[None, None],
                size=(h, w), mode="bilinear", align_corners=False
            ).squeeze().numpy().astype(np.float32)
            acc[vidx][y:y+h, x:x+w] += tile_hw * win
            wgt[vidx][y:y+h, x:x+w] += win

    final = []
    for i in range(num_val):
        m = np.divide(acc[i], np.maximum(wgt[i], 1e-6)).astype(np.float32)
        final.append(m)
    return final


# ---------- core: FAPM per UN tile ----------
def run_fapm_for_tile(train_loader: DataLoader,
                      val_loader: DataLoader,
                      device: torch.device) -> Dict[str, object]:
    """
    Esegue FAPM su UN rettangolo di tile (224x224) per tutto il train/val:
    - costruisce near/far memory (build_memory_payload) sui GOOD di train,
    - calcola heatmap e image_score per tutti i tile di validation,
    - restituisce lista heatmap 224x224 per tile + mapping idx validation globale.
    """

    # Backbone STPM (repo-faithful)
    model = STPM(device=device).to(device).eval()

    # --- costruzione memory per questo tile ---
    mem_payload = build_memory_payload(device, model, train_loader)

    # --- VALIDATION tile-level ---
    score_map_list: List[np.ndarray] = []
    img_scores: List[float] = []
    gt_list: List[int] = []
    order_val_idx: List[int] = []

    with torch.inference_mode():
        for (x, y, m, idx) in tqdm(val_loader, desc="[FAPM-tile] validation", leave=False):
            # x: (B,3,224,224)
            if isinstance(y, torch.Tensor):
                gt_list.extend(y.cpu().numpy().tolist())
            else:
                gt_list.extend(list(y))

            # mapping idx globale validation
            if isinstance(idx, torch.Tensor):
                order_val_idx.extend(idx.cpu().numpy().tolist())
            else:
                order_val_idx.extend([int(i_) for i_ in idx])

            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            for b in range(B):
                amap, s = inference_single_image(device, model, x[b:b+1], mem_payload)
                score_map_list.append(amap)    # 224x224 (già blur)
                img_scores.append(float(s))

            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    gt_np = np.asarray(gt_list, dtype=np.int32)
    img_scores_np = np.asarray(img_scores, dtype=np.float32)

    return {
        "score_map_list": score_map_list,   # heatmap 224x224 per questo tile
        "img_scores": img_scores_np,        # image-level per questo tile (ordine val_loader)
        "gt_np": gt_np,
        "order_val_idx": order_val_idx,     # mapping indice locale -> indice globale validation
    }


# ---------- MAIN ----------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

    # Dataset base SENZA crop/resize globale: SOLO full-res
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=None,                # full-res
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=TEST_SEED,          
        train_seed=TRAIN_SEED,
        transform=None,
        rgb_policy="fullres_only",    # come per SPADE/PaDiM tiled
    )
    print("[meta]", meta)

    # HxW dal dataset (risoluzione full-res)
    if AUTO_SIZE_FROM_DATA:
        sample_img, _, _ = train_set[0]
        _, H0, W0 = sample_img.shape
        TGT_H, TGT_W = int(H0), int(W0)
    else:
        raise RuntimeError("AUTO_SIZE_FROM_DATA=False non gestito qui")

    # Griglia a misura fissa 224x224 (sliding)
    grid = compute_tile_grid_by_size(
        TGT_H, TGT_W,
        tile_h=BACKBONE_IMG_SIZE,
        tile_w=BACKBONE_IMG_SIZE,
        overlap=FIXED_OVERLAP
    )
    N_tiles = len(grid)
    print("[grid] {} tile fissi 224x224, overlap={} -> esempio primi 3: {}".format(
        N_tiles, FIXED_OVERLAP, grid[:3]
    ))

    tile_to_rect: Dict[int, Tuple[int,int,int,int]] = {i: r for i, r in enumerate(grid)}
    tile_val_heatmaps: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(N_tiles)}
    tile_img_scores: Dict[int, Dict[int, float]] = {i: {} for i in range(N_tiles)}

    gt_global = None

    # ====== Loop sequenziale sui tile ======
    for t_id, rect in enumerate(grid):
        print("\n=== TILE {}/{} rect={} ===".format(t_id+1, N_tiles, rect))

        train_tile_ds = TileViewDatasetTrain(train_set, rect, out_size=BACKBONE_IMG_SIZE)
        val_tile_ds   = TileViewDatasetVal(val_set,   rect, out_size=BACKBONE_IMG_SIZE)

        pin = (device.type == "cuda")
        # puoi aumentare/diminuire batch_size in base alla GPU
        train_loader = DataLoader(train_tile_ds, batch_size=16, shuffle=False,
                                  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_tile_ds,   batch_size=8, shuffle=False,
                                  num_workers=0, pin_memory=pin)

        out = run_fapm_for_tile(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        order_val_idx: List[int]         = out["order_val_idx"]
        score_map_list: List[np.ndarray] = out["score_map_list"]
        img_scores_tile: np.ndarray      = out["img_scores"]
        gt_np_tile: np.ndarray           = out["gt_np"]

        if gt_global is None:
            gt_global = gt_np_tile

        # heatmap per-tile e image-score in RAM (nessun file per-tile)
        for local_i, base_idx in enumerate(order_val_idx):
            tile_val_heatmaps[t_id][base_idx] = score_map_list[local_i]
            tile_img_scores[t_id][base_idx]   = float(img_scores_tile[local_i])

        # cleanup
        del train_loader, val_loader, train_tile_ds, val_tile_ds
        del out, score_map_list, order_val_idx, img_scores_tile, gt_np_tile
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ====== RICOMPOSIZIONE & IMAGE-LEVEL AGGREGATO ======
    if DO_RECOMPOSE:
        print("\n[recompose] Ricompongo le FAPM heatmap full-res dal validation set...")
        num_val = len(val_set)

        # full-res heatmaps HxW
        full_res_maps = recompose_full_heatmaps(
            num_val=num_val,
            tile_heatmaps=tile_val_heatmaps,
            tile_to_rect=tile_to_rect,
            out_H=TGT_H, out_W=TGT_W
        )

        # image-level aggregato dai tile (media o max sugli image_score dei tile)
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

        # valutazione stile FAPM, ma su full-res:
        results = run_pixel_level_evaluation(
            score_map_list=full_res_maps,
            val_set=val_set,
            img_scores=agg_scores,
            use_threshold="pro",
            fpr_limit=FPR_LIMIT,
            vis=True,
            vis_ds_or_loader=val_set
        )
        print_pixel_report(results, title="{} | {}  tiles={}  agg={}".format(
            METHOD, CODICE_PEZZO, N_tiles, IMG_SCORE_AGG
        ))

        if SAVE_FINAL_FULLRES:
            np.savez_compressed(
                os.path.join(SAVE_DIR, "{}_fapm_fullres_val_heatmaps.npz".format(CODICE_PEZZO)),
                **{str(i): m for i, m in enumerate(full_res_maps)}
            )
            print("[save] full-res FAPM heatmaps salvate (pacchetto unico).")

    print("\n[done] FAPM tiled completato.")


if __name__ == "__main__":
    #main()
    seed_to_try = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for seed in seed_to_try:
        TRAIN_SEED = seed
        print("----- TRAIN_SEED:", TRAIN_SEED, "| TEST_SEED:", TEST_SEED)
        main()