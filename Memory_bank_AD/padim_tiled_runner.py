# padim_tiled_runner.py
# PaDiM "divide & conquer" sequenziale su tile 224x224 (sliding grid, overlap opzionale).
# Ricomposizione full-res con cosine window + image-level aggregato via max su mappa normalizzata.

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
METHOD = "PADIM_TILED"
CODICE_PEZZO = "PZ3"

# Posizioni "good" per il TRAIN (feature bank)
TRAIN_POSITIONS = ["pos2"]

# Quanti GOOD per posizione spostare in VALIDATION (ed escludere dal TRAIN)
VAL_GOOD_PER_POS = 0

# Da quali posizioni prendere GOOD e FAULT per la VALIDATION
VAL_GOOD_SCOPE  = ["pos2"]
VAL_FAULT_SCOPE = ["pos2"]

# Percentuale di GOOD (dopo il taglio per la val) da usare nel TRAIN
GOOD_FRACTION = 1.0

SEED = 42

# PaDiM
PADIM_D   = 550          # canali selezionati (<= C_total)
RIDGE     = 0.01         # stabilizzazione cov
BACKBONE_IMG_SIZE = 224  # dimensione tile / input backbone

# --- Griglia: MISURA FISSA 224x224 (sliding grid) ---
FIXED_OVERLAP = 0        # overlap opzionale (0, 16, 32, ...)
# Nota: stride = 224 - FIXED_OVERLAP; copertura totale con clamp ai bordi.

# Visual/valutazione
GAUSSIAN_SIGMA = 4       # smoothing sulle heatmap per-tile
FPR_LIMIT = 0.01
SAVE_DIR = "./_padim_tiled_outputs"
DO_RECOMPOSE = True
AUTO_SIZE_FROM_DATA = True  # usa HxW dal dataset full-res

# Salvataggi
SAVE_FINAL_FULLRES = True   # salva SOLO pacchetto finale full-res (nessun file per-tile)
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


# ----- util PaDiM -----
def embedding_concat_nn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Allineamento PaDiM via nearest-neighbor (replica a blocchi s×s) + concat canali.
    x: (B, C1, H1, W1), y: (B, C2, H2, W2), con H1/H2 intero.
    out: (B, C1+C2, H1, W1)
    """
    y_up = F.interpolate(y, size=(x.shape[-2], x.shape[-1]), mode='nearest')
    return torch.cat([x, y_up], dim=1)


# ---------- core: PaDiM per UN tile ----------
def run_padim_for_tile(train_loader: DataLoader,
                       val_loader: DataLoader,
                       device: torch.device,
                       padim_d: int = PADIM_D,
                       ridge: float = RIDGE,
                       gaussian_sigma: int = GAUSSIAN_SIGMA) -> Dict[str, object]:
    """
    Esegue PaDiM su UN rettangolo di tile (224x224) per tutto il train/val:
    - stima mean/cov in streaming sui GOOD di train (due passate),
    - calcola heatmap Mahalanobis per tutti i tile di validation (224x224),
    - restituisce lista heatmap per tile + mapping idx validation globale.
    """

    # ======== BACKBONE + HOOK ========
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model   = wide_resnet50_2(weights=weights).to(device)
    model.eval()

    outputs: List[torch.Tensor] = []
    def hook(_m, _in, out): outputs.append(out)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    rng = torch.Generator().manual_seed(1024)
    sel_idx = None
    d = None
    H = W = L = None

    # ---------- PASSO 1: stima MEDIA in streaming ----------
    N = 0
    sum_x = None  # tensor (d, L)

    with torch.inference_mode():
        for (x, _, _, _) in tqdm(train_loader, desc="[PaDiM tile] pass1 mean", leave=False):
            x = x.to(device, non_blocking=True)
            _ = model(x)
            l1, l2, l3 = [t.cpu() for t in outputs[:3]]
            outputs.clear()

            emb_b = embedding_concat_nn(l1, l2)
            emb_b = embedding_concat_nn(emb_b, l3)          # (B, C_total, Hf, Wf)
            del l1, l2, l3

            if sel_idx is None:
                C_total = emb_b.shape[1]
                d = int(min(padim_d, C_total))
                sel_idx = torch.randperm(C_total, generator=rng)[:d].tolist()

            emb_b = emb_b[:, sel_idx, :, :]                # (B, d, Hf, Wf)
            B = emb_b.shape[0]
            H, W = emb_b.shape[-2], emb_b.shape[-1]
            L = H * W

            E = emb_b.view(B, d, L).to(torch.float32)      # (B, d, L)

            if sum_x is None:
                sum_x = E.sum(dim=0)                       # (d, L)
            else:
                sum_x += E.sum(dim=0)
            N += B

            del emb_b, E, x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    mean = (sum_x / float(N)).cpu().numpy().astype(np.float32)  # (d, L)
    del sum_x

    # ---------- PASSO 2: stima COVARIANZA in streaming + tiling su L ----------
    cov = np.zeros((d, d, L), dtype=np.float32)
    TILE = 256  # tiling sugli indici spaziali della feature map

    with torch.inference_mode():
        for (x, _, _, _) in tqdm(train_loader, desc="[PaDiM tile] pass2 cov", leave=False):
            x = x.to(device, non_blocking=True)
            _ = model(x)
            l1, l2, l3 = [t.cpu() for t in outputs[:3]]
            outputs.clear()

            emb_b = embedding_concat_nn(l1, l2)
            emb_b = embedding_concat_nn(emb_b, l3)          # (B, C_total, Hf, Wf)
            emb_b = emb_b[:, sel_idx, :, :]                # (B, d, Hf, Wf)
            B = emb_b.shape[0]
            E = emb_b.view(B, d, L).numpy().astype(np.float32)   # (B, d, L)
            del l1, l2, l3, emb_b

            for l0 in range(0, L, TILE):
                l1_ = min(l0 + TILE, L)
                t = l1_ - l0

                diffs = E[:, :, l0:l1_] - mean[:, l0:l1_][None, :, :]   # (B, d, t)
                cov[:, :, l0:l1_] += np.einsum('bdt,bkt->dkt', diffs, diffs, optimize=True).astype(np.float32)

            del E, x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    cov /= float(max(1, N - 1))
    cov += (ridge * np.eye(d, dtype=np.float32))[:, :, None]  # (d,d,L)

    # ======== VALIDATION (tile-level) ========
    score_map_list: List[np.ndarray] = []
    gt_list: List[int] = []
    order_val_idx: List[int] = []

    with torch.inference_mode():
        for (x, y, m, idx) in tqdm(val_loader, desc="[PaDiM tile] validation", leave=False):
            # gt e mapping indice globale dell'immagine di validation
            if isinstance(y, torch.Tensor):
                gt_list.extend(y.cpu().numpy().tolist())
            else:
                gt_list.extend(list(y))
            if isinstance(idx, torch.Tensor):
                order_val_idx.extend(idx.cpu().numpy().tolist())
            else:
                order_val_idx.extend([int(idx_) for idx_ in idx])

            x = x.to(device, non_blocking=True)
            _ = model(x)
            l1, l2, l3 = [t.cpu().detach() for t in outputs[:3]]
            outputs.clear()

            emb_t = embedding_concat_nn(l1, l2)
            emb_t = embedding_concat_nn(emb_t, l3)         # (Bv, Ctot, Hf, Wf)
            del l1, l2, l3

            idx_ch = torch.tensor(sel_idx, dtype=torch.long)
            emb_t = torch.index_select(emb_t, 1, idx_ch)   # (Bv, d, Hf, Wf)
            Bv, dv, Hc, Wc = emb_t.shape
            assert (Hc, Wc) == (H, W)

            emb_np_v = emb_t.view(Bv, dv, Hc * Wc).numpy().astype(np.float32)  # (Bv, d, L)
            mean_v   = mean                                                    # (d, L)
            cov_v    = cov                                                     # (d, d, L)

            Lloc = Hc * Wc
            TILE2 = 256
            dist2_LB = np.empty((Lloc, Bv), dtype=np.float32)

            for l0 in range(0, Lloc, TILE2):
                l1_ = min(l0 + TILE2, Lloc)
                t = l1_ - l0

                # diffs_t: (t, Bv, d)
                diffs_t = np.transpose(emb_np_v[:, :, l0:l1_], (2, 0, 1)) - mean_v[:, l0:l1_].T[:, None, :]

                cov_t = np.transpose(cov_v[:, :, l0:l1_], (2, 0, 1)).copy()  # (t, d, d)
                eps = 1e-2
                cov_t += eps * np.eye(dv, dtype=np.float32)[None, :, :]

                cov_t_t = torch.from_numpy(cov_t)           # (t,d,d)
                diffs_t_t = torch.from_numpy(diffs_t)       # (t,Bv,d)

                Lfac = torch.linalg.cholesky(cov_t_t)       # (t,d,d)
                diffsT = diffs_t_t.transpose(1, 2).contiguous()  # (t,d,Bv)
                sol = torch.cholesky_solve(diffsT, Lfac)         # (t,d,Bv)

                dist2_tB = (diffsT * sol).sum(dim=1)             # (t,Bv)
                dist2_LB[l0:l1_, :] = dist2_tB.cpu().numpy().astype(np.float32)

            dist_arr = np.sqrt(dist2_LB.T).astype(np.float32).reshape(Bv, Hc, Wc)  # (Bv,Hf,Wf)

            # upsample a 224x224 + gaussian smoothing
            dist_t  = torch.from_numpy(dist_arr).unsqueeze(1)  # (Bv,1,Hf,Wf)
            score_b = F.interpolate(dist_t, size=BACKBONE_IMG_SIZE, mode='bilinear',
                                    align_corners=False).squeeze(1).numpy()
            for i in range(score_b.shape[0]):
                if gaussian_sigma > 0:
                    score_b[i] = gaussian_filter(score_b[i], sigma=gaussian_sigma)

            # accumulo heatmap per-tile (224x224) per ciascuna immagine del batch
            for i in range(score_b.shape[0]):
                score_map_list.append(score_b[i])

            del emb_t, emb_np_v, dist_t, dist_arr, score_b, x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    gt_np = np.asarray(gt_list, dtype=np.int32)

    return {
        "score_map_list": score_map_list,   # heatmap 224x224 per questo tile
        "gt_np": gt_np,
        "order_val_idx": order_val_idx,     # mapping indice locale -> indice globale validation
    }


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


# ---------- MAIN ----------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

    # Dataset base SENZA crop/resize globale: SOLO full-res
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

    # HxW dal dataset (risoluzione full-res)
    if AUTO_SIZE_FROM_DATA:
        sample_img, _, _ = train_set[0]
        _, H0, W0 = sample_img.shape
        TGT_H, TGT_W = int(H0), int(W0)
    else:
        raise RuntimeError("AUTO_SIZE_FROM_DATA=False non gestito qui")

    # Griglia a misura fissa 224x224 (sliding)
    grid = compute_tile_grid_by_size(TGT_H, TGT_W,
                                     tile_h=BACKBONE_IMG_SIZE,
                                     tile_w=BACKBONE_IMG_SIZE,
                                     overlap=FIXED_OVERLAP)
    N_tiles = len(grid)
    print("[grid] {} tile fissi 224x224, overlap={} -> esempio primi 3: {}".format(
        N_tiles, FIXED_OVERLAP, grid[:3]
    ))

    tile_to_rect: Dict[int, Tuple[int,int,int,int]] = {i: r for i, r in enumerate(grid)}
    tile_val_heatmaps: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(N_tiles)}

    # Per le label (uguali per tutti i tile) – prendiamo quelle dal primo tile
    gt_global = None

    # ====== Loop sequenziale sui tile ======
    for t_id, rect in enumerate(grid):
        print("\n=== TILE {}/{} rect={} ===".format(t_id+1, N_tiles, rect))

        train_tile_ds = TileViewDataset(train_set, rect, out_size=BACKBONE_IMG_SIZE)
        val_tile_ds   = TileViewDataset(val_set,   rect, out_size=BACKBONE_IMG_SIZE)

        pin = (device.type == "cuda")
        train_loader = DataLoader(train_tile_ds, batch_size=32, shuffle=False,
                                  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_tile_ds,   batch_size=32, shuffle=False,
                                  num_workers=0, pin_memory=pin)

        out = run_padim_for_tile(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            padim_d=PADIM_D,
            ridge=RIDGE,
            gaussian_sigma=GAUSSIAN_SIGMA
        )

        order_val_idx: List[int]          = out["order_val_idx"]
        score_map_list: List[np.ndarray]  = out["score_map_list"]
        gt_np_tile: np.ndarray            = out["gt_np"]

        if gt_global is None:
            gt_global = gt_np_tile  # uguale per tutti i tile (stesso val_set)

        # heatmap per-tile in RAM (senza scrivere su disco)
        for local_i, base_idx in enumerate(order_val_idx):
            tile_val_heatmaps[t_id][base_idx] = score_map_list[local_i]

        # cleanup
        del train_loader, val_loader, train_tile_ds, val_tile_ds, out, score_map_list, order_val_idx, gt_np_tile
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ====== RICOMPOSIZIONE & IMAGE-LEVEL ======
    if DO_RECOMPOSE:
        print("\n[recompose] Ricompongo le heatmap full-res dal validation set...")
        num_val = len(val_set)

        # full-res heatmaps HxW
        full_res_maps = recompose_full_heatmaps(
            num_val=num_val,
            tile_heatmaps=tile_val_heatmaps,
            tile_to_rect=tile_to_rect,
            out_H=TGT_H, out_W=TGT_W
        )

        # raw_score_maps = heatmap RAW full-res (prima di normalizzazione globale)
        raw_score_maps = np.asarray(full_res_maps, dtype=np.float32)   # (N, H, W)
        smax, smin = raw_score_maps.max(), raw_score_maps.min()
        scores_norm = (raw_score_maps - smin) / (smax - smin + 1e-12)  # (N,H,W) SOLO per image score

        img_scores_list = scores_norm.reshape(scores_norm.shape[0], -1).max(axis=1)

        # valutazione stile PaDiM (PRO pixel-level + image-level da img_scores_list)
        results = run_pixel_level_evaluation(
            score_map_list=list(raw_score_maps),   # heatmap RAW full-res
            val_set=val_set,
            img_scores=img_scores_list,
            use_threshold="pro",
            fpr_limit=FPR_LIMIT,
            vis=True,
            vis_ds_or_loader=val_set
        )
        print_pixel_report(results, title="{} | {}  tiles={}  overlap={}".format(
            METHOD, CODICE_PEZZO, N_tiles, FIXED_OVERLAP
        ))

        if SAVE_FINAL_FULLRES:
            np.savez_compressed(os.path.join(SAVE_DIR, "{}_padim_fullres_val_heatmaps.npz".format(CODICE_PEZZO)),
                                **{str(i): m for i, m in enumerate(full_res_maps)})
            print("[save] full-res PaDiM heatmaps salvate (pacchetto unico).")

    print("\n[done] PaDiM tiled completato.")


if __name__ == "__main__":
    main()
