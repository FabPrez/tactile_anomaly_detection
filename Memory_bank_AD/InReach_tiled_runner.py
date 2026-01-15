# InReach_tiled_runner.py
# InReaCh "divide & conquer" sequenziale su tile 224x224 (sliding grid, overlap opzionale).
# Ricomposizione full-res con cosine window + image-level aggregato (mean/max) dai tile.

import os, gc, random
from typing import List, Tuple, Dict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- importa classi/funzioni dal tuo InReach.py ----
from InReach import (
    InReaCh,
    load_wide_resnet_50,
    super_seed,
    ASSOC_DEPTH,
    MIN_CHANNEL_LENGTH,
    MAX_CHANNEL_STD,
    FILTER_SIZE,
    POS_EMBED_WEIGHT_ON,   # fattore di scala per PE
)

# --- tue utility (progetto) ---
from data_loader import build_ad_datasets
from ad_analysis import run_pixel_level_evaluation, print_pixel_report


# ----------------- CONFIG -----------------
METHOD = "INREACH_TILED_RUNNER"
CODICE_PEZZO = "PZ3"

TRAIN_POSITIONS = ["pos1"]
VAL_GOOD_PER_POS = 20
VAL_GOOD_SCOPE   = ["pos1"]
VAL_FAULT_SCOPE  = ["pos1"]
GOOD_FRACTION    = 0.2

TEST_SEED  = 42  # controlla *solo* la scelta delle immagini di validation/test
TRAIN_SEED = 42  # lo puoi cambiare tu per variare

# dimensione tile / input backbone
BACKBONE_IMG_SIZE = 224

# --- Griglia: MISURA FISSA 224x224 (sliding grid) ---
FIXED_OVERLAP = 0        # overlap opzionale (0, 16, 32, ...)

# Valutazione
FPR_LIMIT = 0.01
SAVE_DIR = "./_inreach_tiled_outputs"
DO_RECOMPOSE = True
AUTO_SIZE_FROM_DATA = True  # usa HxW dal dataset full-res

# Salvataggi
SAVE_FINAL_FULLRES = True   # salva SOLO pacchetto finale full-res (nessun file per-tile)

# Aggregazione degli image-level score dai tile: "mean" oppure "max"
IMG_SCORE_AGG = "max"

# Per evitare rotazioni/allineamenti strani sui tile:
# soglia molto bassa => il gate non attiva mai l'allineamento, ma tiene le PE attive.
POS_EMBED_THRESH_TILED = -1.0
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


# ---------- helper: estrazione tile HWC uint8 ----------
def extract_tile_uint8(img_t: torch.Tensor,
                       rect: Tuple[int,int,int,int],
                       out_size: int = 224) -> np.ndarray:
    """
    img_t: tensor (C,H,W) float [0,1]
    rect: (y,x,h,w) in coordinate full-res
    Ritorna patch (out_size,out_size,3) uint8 [0..255] per InReaCh.
    """
    C, H, W = img_t.shape
    y, x, h, w = rect
    y = max(0, min(y, H-1))
    x = max(0, min(x, W-1))
    h = max(1, min(h, H - y))
    w = max(1, min(w, W - x))

    patch = img_t[:, y:y+h, x:x+w]        # (C,h,w)
    patch = patch.unsqueeze(0)            # (1,C,h,w)
    patch = F.interpolate(patch, size=(out_size, out_size),
                          mode="bilinear", align_corners=False).squeeze(0)  # (C,out,out)

    patch_np = patch.permute(1,2,0).cpu().numpy()  # (H,W,C) float [0,1]
    patch_np = np.clip(patch_np * 255.0, 0, 255).astype(np.uint8)
    return patch_np


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


# ---------- core: InReaCh per UN tile ----------
def run_inreach_for_tile(train_imgs_tile: List[np.ndarray],
                         val_imgs_tile: List[np.ndarray],
                         base_model) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Esegue InReaCh su UN rettangolo di tile (224x224) per tutto il train/val:
    - costruisce i canali / indice FAISS sui GOOD di train,
    - calcola le heatmap per tutti i tile di validation,
    - ritorna:
        score_maps_tile: lista di heatmap 2D 224x224
        img_scores_tile: array (N_val,) con max score per tile
    """
    if len(train_imgs_tile) == 0:
        raise RuntimeError("run_inreach_for_tile: nessuna immagine GOOD per questo tile.")

    # InReaCh con PE-gate (ma soglia molto bassa => niente rotazioni di allineamento sui tile)
    inreach = InReaCh(
        images=train_imgs_tile,
        model=base_model,
        assoc_depth=ASSOC_DEPTH,
        min_channel_length=MIN_CHANNEL_LENGTH,
        max_channel_std=MAX_CHANNEL_STD,
        filter_size=FILTER_SIZE,
        pos_embed_thresh=POS_EMBED_THRESH_TILED,
        pos_embed_weight=POS_EMBED_WEIGHT_ON,
        quite=True,
    )

    score_maps, _ = inreach.predict(val_imgs_tile, t_masks=None, quite=True)

    img_scores_tile = np.asarray([float(np.max(s)) for s in score_maps], dtype=np.float32)
    return score_maps, img_scores_tile


# ---------- MAIN ----------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    super_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

    # Dataset base SENZA resize globale: immagini full-res
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=None,                  # full-res
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=TEST_SEED,          
        train_seed=TRAIN_SEED,
        transform=None,
        rgb_policy="fullres_only",      # come SPADE/PaDiM/FAPM tiled
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

    # Label globali per la validation
    gt_list = []
    for idx in range(len(val_set)):
        _, lbl, _ = val_set[idx]
        lbl_int = int(lbl.item()) if isinstance(lbl, torch.Tensor) else int(lbl)
        gt_list.append(lbl_int)
    gt_np = np.asarray(gt_list, dtype=np.int32)

    # === modello backbone condiviso per tutti i tile (come nella main originale) ===
    return_nodes = {
        'layer1.0.relu_2': 'Level_1',
        'layer1.1.relu_2': 'Level_2',
        'layer1.2.relu_2': 'Level_3',
        'layer2.0.relu_2': 'Level_4',
        'layer2.1.relu_2': 'Level_5',
        'layer2.2.relu_2': 'Level_6',
        'layer2.3.relu_2': 'Level_7',
        'layer3.1.relu_2': 'Level_8',
        'layer3.2.relu_2': 'Level_9',
        'layer3.3.relu_2': 'Level_10',
        'layer3.4.relu_2': 'Level_11',
        'layer3.5.relu_2': 'Level_12',
        'layer4.0.relu_2': 'Level_13',
    }
    base_model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)

    # ====== Loop sequenziale sui tile ======
    for t_id, rect in enumerate(grid):
        print("\n=== TILE {}/{} rect={} ===".format(t_id+1, N_tiles, rect))

        # --- costruiamo le immagini GOOD di train per questo tile ---
        train_imgs_tile: List[np.ndarray] = []
        for i in range(len(train_set)):
            img_t, lbl, _ = train_set[i]      # img_t: (C,H,W) float[0,1]
            lbl_int = int(lbl.item()) if isinstance(lbl, torch.Tensor) else int(lbl)
            if lbl_int != 0:
                continue  # InReaCh usa solo GOOD come memoria nominale
            img_tile = extract_tile_uint8(img_t, rect, out_size=BACKBONE_IMG_SIZE)
            train_imgs_tile.append(img_tile)

        if len(train_imgs_tile) == 0:
            print(f"[warning] Nessun GOOD nel train per tile {t_id} -> salto questo tile.")
            continue

        # --- immagini di validation (tutte) per questo tile, in ordine fisso [0..N_val-1] ---
        val_imgs_tile: List[np.ndarray] = []
        for idx in range(len(val_set)):
            img_t, _, _ = val_set[idx]
            img_tile = extract_tile_uint8(img_t, rect, out_size=BACKBONE_IMG_SIZE)
            val_imgs_tile.append(img_tile)

        # --- esegui InReaCh su questo tile ---
        score_map_list, img_scores_tile = run_inreach_for_tile(
            train_imgs_tile=train_imgs_tile,
            val_imgs_tile=val_imgs_tile,
            base_model=base_model,
        )

        # score_map_list e img_scores_tile sono allineati con l'ordine di val_imgs_tile,
        # che è l'ordine naturale del val_set: idx = 0..N_val-1
        num_val = len(val_set)
        assert len(score_map_list) == num_val

        for base_idx in range(num_val):
            tile_val_heatmaps[t_id][base_idx] = score_map_list[base_idx]
            tile_img_scores[t_id][base_idx]   = float(img_scores_tile[base_idx])

        # cleanup parziale
        del train_imgs_tile, val_imgs_tile, score_map_list, img_scores_tile
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ====== RICOMPOSIZIONE & IMAGE-LEVEL AGGREGATO ======
    if DO_RECOMPOSE:
        print("\n[recompose] Ricompongo le InReaCh heatmap full-res dal validation set...")
        num_val = len(val_set)

        # full-res heatmaps HxW
        full_res_maps = recompose_full_heatmaps(
            num_val=num_val,
            tile_heatmaps=tile_val_heatmaps,
            tile_to_rect=tile_to_rect,
            out_H=TGT_H, out_W=TGT_W
        )

        # image-level aggregato dai tile (media o max sugli score dei tile)
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

        # valutazione stile InReaCh, ma su full-res tiled
        results = run_pixel_level_evaluation(
            score_map_list=full_res_maps,
            val_set=val_set,
            img_scores=agg_scores,
            use_threshold="pro",
            fpr_limit=FPR_LIMIT,
            vis=False,
            vis_ds_or_loader=val_set
        )
        print_pixel_report(results, title="{} | {}  tiles={}  agg={}".format(
            METHOD, CODICE_PEZZO, N_tiles, IMG_SCORE_AGG
        ))

        if SAVE_FINAL_FULLRES:
            np.savez_compressed(
                os.path.join(SAVE_DIR, "{}_inreach_fullres_val_heatmaps.npz".format(CODICE_PEZZO)),
                **{str(i): m for i, m in enumerate(full_res_maps)}
            )
            print("[save] full-res InReaCh heatmaps salvate (pacchetto unico).")

    print("\n[done] InReaCh tiled completato.")


if __name__ == "__main__":
    #main()

    seed_to_try = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for seed in seed_to_try:
        TRAIN_SEED = seed
        print("----- TRAIN_SEED:", TRAIN_SEED, "| TEST_SEED:", TEST_SEED)
        main()
