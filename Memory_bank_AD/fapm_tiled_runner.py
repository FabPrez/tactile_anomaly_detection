# fapm_tiled_runner.py
# FAPM "divide & conquer" sequenziale su N tile per immagine (solo full-res RGB).
# - Nessun center-crop/pre-crop: dataset base full-res (rgb_policy="fullres_only")
# - Per ogni tile: estrazione feature (layer1..3), memory bank FAPM adattiva (per patch),
#   matching co-locato -> heatmap validation
# - Salva SOLO heatmap per-tile; poi ricompone full-res e valuta (pixel- & image-level)

import os, gc, json
from contextlib import nullcontext
from typing import List, Tuple, Dict
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# tuoi moduli
from data_loader import build_ad_datasets
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# --------------- CONFIG ---------------
METHOD = "FAPM_TILED"
CODICE_PEZZO = "PZ3"

# split
TRAIN_POSITIONS   = ["pos2"]
VAL_GOOD_PER_POS  = 20
VAL_GOOD_SCOPE    = ["pos2"]
VAL_FAULT_SCOPE   = ["pos2"]
GOOD_FRACTION     = 0.30
SEED              = 42

# backbone / feature size
BACKBONE_IMG_SIZE = 224
GAUSSIAN_SIGMA    = 3

# FAPM settings
LAYER_NAMES       = ("layer1","layer2","layer3")
L2NORMALIZE       = True
MEM_AVG_PER_LOC   = 4
MEM_MIN_PER_LOC   = 1
MEM_MAX_PER_LOC   = 8
K_PATCH           = 3
DEVICE_CDIST      = "cuda" if torch.cuda.is_available() else "cpu"

# Image-level pooling
IMG_SCORE_POOL    = "max"  # "max" | "p99" | "mean"
FPR_LIMIT         = 0.01

# tiling (griglia a conteggio fisso)
USE_FIXED_GRID_COUNTS = True
FIXED_ROWS = 2
FIXED_COLS = 2
FIXED_OVERLAP = 0

# alternativa stride-based (se metti USE_FIXED_GRID_COUNTS=False)
TILE_W, TILE_H = 384, 384
OVERLAP = 64

# IO / run
SAVE_DIR          = "./_fapm_tiled_outputs"
DO_RECOMPOSE      = True
AUTO_SIZE_FROM_DATA = True
BATCH_SIZE        = 32
# --------------------------------------


# ----------- utilities: grid -----------
def compute_tile_grid(H: int, W: int, tile_h: int, tile_w: int, overlap: int):
    sh = max(1, tile_h - overlap); sw = max(1, tile_w - overlap)
    ys = list(range(0, max(1, H - tile_h + 1), sh))
    xs = list(range(0, max(1, W - tile_w + 1), sw))
    if ys[-1] + tile_h < H: ys.append(H - tile_h)
    if xs[-1] + tile_w < W: xs.append(W - tile_w)
    rects = []
    for y in ys:
        for x in xs:
            rects.append((y, x, min(tile_h, H - y), min(tile_w, W - x)))
    return rects  # (y,x,h,w)

def compute_tile_grid_by_counts(H: int, W: int, n_rows: int, n_cols: int, overlap: int = 0):
    ys = np.linspace(0, H, n_rows + 1, dtype=int)
    xs = np.linspace(0, W, n_cols + 1, dtype=int)
    rects = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0,y1 = ys[r], ys[r+1]
            x0,x1 = xs[c], xs[c+1]
            if overlap > 0:
                if r > 0:        y0 = max(0, y0 - overlap//2)
                if r < n_rows-1: y1 = min(H, y1 + overlap//2)
                if c > 0:        x0 = max(0, x0 - overlap//2)
                if c < n_cols-1: x1 = min(W, x1 + overlap//2)
            h = max(1, y1 - y0); w = max(1, x1 - x0)
            rects.append((int(y0), int(x0), int(h), int(w)))
    return rects


# --------- dataset wrapper: tile view (accetta CHW tensors dal dataset base) ---------
class TileViewDataset(Dataset):
    def __init__(self, base_ds: Dataset, tile_rect: Tuple[int,int,int,int], out_size: int = 224):
        self.base = base_ds
        self.y, self.x, self.h, self.w = tile_rect
        self.out = out_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img_t, lbl, mask_t = self.base[i]     # img_t: (C,H,W) float[0,1]
        C,H,W = img_t.shape
        y,x,h,w = self.y, self.x, self.h, self.w
        y = max(0, min(y, H-1)); x = max(0, min(x, W-1))
        h = max(1, min(h, H - y)); w = max(1, min(w, W - x))
        img_tile = img_t[:, y:y+h, x:x+w]
        img_tile = F.interpolate(img_tile.unsqueeze(0), size=(self.out, self.out),
                                 mode='bilinear', align_corners=False).squeeze(0)
        img_tile = (img_tile - self.mean) / self.std
        if mask_t is not None:
            m = mask_t[y:y+h, x:x+w].float().unsqueeze(0).unsqueeze(0)
            m = F.interpolate(m, size=(self.out, self.out), mode="nearest").squeeze().cpu().numpy().astype(np.float32)
        else:
            m = None
        return img_tile, int(lbl.item() if isinstance(lbl, torch.Tensor) else int(lbl)), m, i


# ---------------- FAPM core: memory & matching ----------------
def set_all_seeds(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def extract_feats_layers(model, loader, device) -> OrderedDict:
    outputs = []
    def hook(_m,_i,o): outputs.append(o)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    out = OrderedDict([(n, []) for n in LAYER_NAMES])
    with torch.inference_mode():
        for x,_,_,_ in tqdm(loader, desc='| feats |'):
            _ = model(x.to(device, non_blocking=True))
            for k, v in zip(out.keys(), outputs):
                out[k].append(v.detach().cpu())
            outputs.clear()
    for k in out: out[k] = torch.cat(out[k], dim=0)  # (N,C,H,W) on CPU
    return out

@torch.no_grad()
def farthest_first_per_cell(X: torch.Tensor, m: int) -> torch.Tensor:
    N = X.size(0)
    if m >= N or N == 0:
        return torch.arange(N, dtype=torch.long)
    sel = [np.random.randint(0, N)]
    dmin = torch.cdist(X[sel], X, p=2.0)[0]
    for _ in range(1, m):
        i = int(torch.argmax(dmin).item())
        sel.append(i)
        dmin = torch.minimum(dmin, torch.cdist(X[i:i+1], X, p=2.0)[0])
    return torch.tensor(sel, dtype=torch.long)

def allocate_adaptive_counts(disp_map: torch.Tensor, avg=MEM_AVG_PER_LOC,
                             mmin=MEM_MIN_PER_LOC, mmax=MEM_MAX_PER_LOC):
    d = disp_map.float()
    mean_d = torch.clamp(d.mean(), min=1e-12)
    m = (d / mean_d) * float(avg)
    m = torch.clamp(torch.round(m), min=mmin, max=mmax).to(torch.long)
    return m

def build_layer_memory_bank(feats_l: torch.Tensor, l2norm_feats=True) -> dict:
    if l2norm_feats:
        feats_l = l2norm(feats_l, dim=1)
    N,C,H,W = feats_l.shape
    feats_l = feats_l.permute(0,2,3,1).contiguous().view(N, H*W, C)  # (N, HW, C)

    std_hw_c = torch.std(feats_l, dim=0, unbiased=False)  # (HW,C)
    disp_hw  = std_hw_c.mean(dim=1).view(H, W)            # (H,W)
    m_hw = allocate_adaptive_counts(disp_hw)              # (H,W)

    mem_chunks = []
    offsets = torch.zeros((H, W, 2), dtype=torch.int64)
    cursor = 0
    for h in range(H):
        for w in range(W):
            X = feats_l[:, h*W + w, :].float()   # (N,C)
            m = int(m_hw[h,w].item())
            idx = farthest_first_per_cell(X, m)
            Ym = X[idx]                          # (m',C)
            mem_chunks.append(Ym)
            start = cursor
            cursor += Ym.shape[0]
            offsets[h,w,0] = start
            offsets[h,w,1] = cursor

    mem = torch.cat(mem_chunks, dim=0).contiguous() if len(mem_chunks) else torch.empty((0, C))
    return {
        "mem": mem, "offsets": offsets, "counts_per_loc": m_hw,
        "C": int(C), "H": int(H), "W": int(W),
        "l2norm": bool(l2norm_feats),
    }

def build_fapm_memory(train_feats: OrderedDict, l2norm_feats=True) -> dict:
    mb = {}
    for lname in LAYER_NAMES:
        print(f"[FAPM] memory for {lname} ...")
        mb[lname] = build_layer_memory_bank(train_feats[lname].contiguous(), l2norm_feats)
        print(f"  -> mem = {mb[lname]['mem'].shape[0]} (C={mb[lname]['C']}, HxW={mb[lname]['H']}x{mb[lname]['W']})")
    return mb

@torch.no_grad()
def infer_score_map_one_image(test_one: OrderedDict, mbank: dict,
                              k_patch=K_PATCH, img_size=BACKBONE_IMG_SIZE, gaussian_sigma=GAUSSIAN_SIGMA):
    layer_maps = []
    device_cdist = torch.device(DEVICE_CDIST)
    for lname in LAYER_NAMES:
        tf = test_one[lname]  # (1,C,H,W) CPU
        if mbank[lname]["l2norm"]:
            tf = l2norm(tf, dim=1)
        _, C, H, W = tf.shape
        assert (H, W) == (mbank[lname]["H"], mbank[lname]["W"])
        q = tf.view(C, H*W).t().contiguous()            # (HW,C) CPU
        mem = mbank[lname]["mem"].to(device_cdist, non_blocking=True)    # (M,C)
        offs = mbank[lname]["offsets"]                  # (H,W,2) CPU

        s_loc = torch.empty((H*W,), dtype=torch.float32)
        for hw in range(H*W):
            h = hw // W; w = hw % W
            a = int(offs[h,w,0].item()); b = int(offs[h,w,1].item())
            if a == b:
                s_loc[hw] = 0.0
                continue
            qi = q[hw:hw+1].to(device_cdist)           # (1,C)
            Mi = mem[a:b]                              # (m,C)
            d = torch.cdist(qi, Mi, p=2.0)             # (1,m)
            if k_patch >= 2 and Mi.size(0) >= k_patch:
                v,_ = torch.topk(d, k=k_patch, dim=1, largest=False)
                s = v.mean(dim=1)[0]
            else:
                s = d.min(dim=1).values[0]
            s_loc[hw] = s.float().cpu()

        m = s_loc.view(1,1,H,W)
        m = F.interpolate(m, size=img_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        if gaussian_sigma and gaussian_sigma > 0:
            m = gaussian_filter(m, sigma=gaussian_sigma)
        layer_maps.append(m)

    fused = np.mean(np.stack(layer_maps, axis=0), axis=0).astype(np.float32)
    return fused


# -------------- ricomposizione --------------
def cosine_window(h, w):
    wy = 0.5*(1 - np.cos(2*np.pi*(np.arange(h)/(h-1)))) if h > 1 else np.ones(1)
    wx = 0.5*(1 - np.cos(2*np.pi*(np.arange(w)/(w-1)))) if w > 1 else np.ones(1)
    return np.outer(wy, wx).astype(np.float32)

def recompose_full_heatmaps(num_val: int,
                            tile_heatmaps: Dict[int, Dict[int, np.ndarray]],
                            tile_to_rect: Dict[int, Tuple[int,int,int,int]],
                            out_H: int, out_W: int) -> List[np.ndarray]:
    acc = [np.zeros((out_H, out_W), np.float32) for _ in range(num_val)]
    wgt = [np.zeros((out_H, out_W), np.float32) for _ in range(num_val)]
    for t_id, rect in tile_to_rect.items():
        y,x,h,w = rect
        win = cosine_window(h, w)
        for vidx, hmap224 in tile_heatmaps[t_id].items():
            tile_hw = F.interpolate(
                torch.from_numpy(hmap224)[None,None],
                size=(h, w), mode="bilinear", align_corners=False
            ).squeeze().numpy().astype(np.float32)
            acc[vidx][y:y+h, x:x+w] += tile_hw * win
            wgt[vidx][y:y+h, x:x+w] += win
    out = []
    for i in range(num_val):
        m = np.divide(acc[i], np.maximum(wgt[i], 1e-6)).astype(np.float32)
        out.append(m)
    return out

def pool_image_score(arr: np.ndarray, mode="max"):
    if mode == "max": return float(np.max(arr))
    if mode == "p99": return float(np.percentile(arr, 99))
    return float(np.mean(arr))


# ---------------------- MAIN ----------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_all_seeds(SEED)

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
        rgb_policy="fullres_only",   # IMPORTANT: tiled => solo cartelle RGB full-res
    )
    print("[meta]", meta)

    # auto size from first sample
    if AUTO_SIZE_FROM_DATA:
        smp,_,_ = train_set[0]
        _, H0, W0 = smp.shape
        TGT_H, TGT_W = int(H0), int(W0)
    else:
        # fallback, imposta manualmente
        TGT_H, TGT_W = 720, 1280

    # griglia
    if USE_FIXED_GRID_COUNTS:
        grid = compute_tile_grid_by_counts(TGT_H, TGT_W, FIXED_ROWS, FIXED_COLS, overlap=FIXED_OVERLAP)
    else:
        grid = compute_tile_grid(TGT_H, TGT_W, TILE_H, TILE_W, OVERLAP)

    N_tiles = len(grid)
    print(f"[grid] {N_tiles} tile -> {grid[:3]}...")

    # backbone
    weights = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model   = wide_resnet50_2(weights=weights).to(device).eval()

    pin = (device.type == "cuda")
    tile_to_rect = {i: r for i,r in enumerate(grid)}
    tile_val_heatmaps: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(N_tiles)}

    # ===== loop sequenziale sui tile =====
    for t_id, rect in enumerate(grid):
        print(f"\n=== TILE {t_id+1}/{N_tiles} rect={rect} ===")

        # loader specifici per tile
        train_tile_ds = TileViewDataset(train_set, rect, out_size=BACKBONE_IMG_SIZE)
        val_tile_ds   = TileViewDataset(val_set,   rect, out_size=BACKBONE_IMG_SIZE)

        train_loader = DataLoader(train_tile_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_tile_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=pin)

        # --- FEATURE CACHE per-tile (in RAM, no pickle per semplicità) ---
        feats_train = extract_feats_layers(model, train_loader, device)  # CPU (N,C,H,W)
        feats_val   = extract_feats_layers(model, val_loader, device)    # CPU (N,C,H,W)

        # --- MEMORY BANK per tile ---
        mbank = build_fapm_memory(feats_train, l2norm_feats=L2NORMALIZE)

        # --- INFERENZA HEATMAP validation per tile ---
        order_val_idx = list(range(feats_val[LAYER_NAMES[0]].shape[0]))
        for i in tqdm(order_val_idx, desc=f"| FAPM infer tile {t_id} |"):
            one = OrderedDict([(ln, feats_val[ln][i:i+1]) for ln in LAYER_NAMES])
            hmap = infer_score_map_one_image(one, mbank,
                                             k_patch=K_PATCH,
                                             img_size=BACKBONE_IMG_SIZE,
                                             gaussian_sigma=GAUSSIAN_SIGMA)
            tile_val_heatmaps[t_id][i] = hmap

        # salva compatti per il tile (npz + rect json)
        np.savez_compressed(os.path.join(SAVE_DIR, f"tile_{t_id:03d}_val_heatmaps.npz"),
                            **{str(k): v for k, v in tile_val_heatmaps[t_id].items()})
        with open(os.path.join(SAVE_DIR, f"tile_{t_id:03d}_rect.json"), "w") as f:
            json.dump({"rect": rect}, f)

        # cleanup forte
        del train_loader, val_loader, train_tile_ds, val_tile_ds
        del feats_train, feats_val, mbank
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ===== ricomposizione + valutazione =====
    if DO_RECOMPOSE:
        print("\n[recompose] Ricompongo le heatmap full-res dal validation set...")
        num_val = len(val_set)
        full_res_maps = recompose_full_heatmaps(
            num_val=num_val,
            tile_heatmaps=tile_val_heatmaps,
            tile_to_rect=tile_to_rect,
            out_H=TGT_H, out_W=TGT_W
        )

        # image-level score (pooling sulle full-res)
        img_scores = np.array([pool_image_score(m, IMG_SCORE_POOL) for m in full_res_maps], dtype=np.float32)

        # valutazione + viewer stile SPADE
        results = run_pixel_level_evaluation(
            score_map_list=full_res_maps,
            val_set=val_set,
            img_scores=img_scores,          # <<< importante: len == len(val_set)
            use_threshold="pro",
            fpr_limit=FPR_LIMIT,
            vis=True,
            vis_ds_or_loader=val_set
        )
        print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO} | tiles={N_tiles}")

        # dump delle full-res heatmaps
        np.savez_compressed(os.path.join(SAVE_DIR, f"{CODICE_PEZZO}_fullres_val_heatmaps.npz"),
                            **{str(i): m for i, m in enumerate(full_res_maps)})

    print("\n[done] FAPM tiled completato. Output in:", SAVE_DIR)


if __name__ == "__main__":
    main()
