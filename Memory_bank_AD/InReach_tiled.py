# InReach_tiled_stream.py
# InReaCh tiled, fedele al paper (Flat L2), con build del bank "streaming" (spill su disco)
# per evitare picchi di RAM. Niente pickle per tile (PERSIST_FEATURES=False in questa variante).

import os, gc, shutil, uuid, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable, Union
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# ================== CONFIG BASE ==================
METHOD        = "INREACH_OFFICIAL_TILED"
CODICE_PEZZO  = "PZ3"

TRAIN_POSITIONS   = ["pos2"]
VAL_GOOD_PER_POS  = 0
VAL_GOOD_SCOPE    = ["pos2"]
VAL_FAULT_SCOPE   = ["pos2"]
GOOD_FRACTION     = 1.0

# tiled a conteggio fisso (es. 2x2)
TILED_ROWS = 2
TILED_COLS = 2
TILED_OVERLAP = 0

# Backbone
IMG_SIZE    = 224
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# I/O cache (per la versione tiled disattivo i pickle per risparmiare spazio/tempo)
PERSIST_FEATURES  = False
PERSIST_BANK      = False       # il bank qui è "lazy" per chunk: meglio NON salvarlo in unico file
PERSIST_CHANNELS  = False

# FAISS
USE_FAISS = True
_FAISS_OK = False
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

# InReaCh iperparametri (fedeli)
FEATURE_LAYER = "layer2"
CORESET_IMGS  = "ALL"     # oppure un intero (8, 16, 32, ...)
SEARCH_RAD    = 1         # 3x3
SIM_MIN       = 0.0
STRIDE_H      = 1
STRIDE_W      = 1
SPAN_MIN      = 0.0
SPREAD_MAX    = float("inf")
BANK_PER_CHANNEL_LIMIT = None

# Performance
BATCH_J   = 32            # immagini train per blocco nella build
USE_AMP   = True          # mixed precision in build
TILE_Q    = 65536         # batch FAISS di query (per immagine)
TILE_B    = 20000         # chunk torch.cdist (se usi il fallback torch)
GAUSS_SIGMA = 2.0

# Spilling
SPILL_ROOT = os.path.join(os.path.dirname(__file__), "_tiled_outputs_inreach")
os.makedirs(SPILL_ROOT, exist_ok=True)

# ================== UTILS ==================
def _p(*a, **k): print(*a, **k)

def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def get_backbone(device):
    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
    m.eval()
    outs = []
    def hook(_m,_i,o): outs.append(o)
    m.layer2[-1].register_forward_hook(hook)  # feature intermedie
    m.avgpool.register_forward_hook(hook)     # globali per coreset
    return m, outs

@torch.no_grad()
def extract_features(model, outs, loader, device):
    Fs, Gs = [], []
    for x,_,_ in tqdm(loader, desc="feature extraction", leave=False):
        _ = model(x.to(device, non_blocking=True))
        layer2, avg = outs[0], outs[1]; outs.clear()
        Fs.append(layer2.detach().cpu())               # (B,C,H,W)
        Gs.append(avg.detach().cpu())                  # (B,2048,1,1)
    F = torch.cat(Fs, 0)                               # (N,C,H,W)
    G = torch.flatten(torch.cat(Gs, 0), 1)             # (N,2048)
    return F, G

@torch.no_grad()
def kcenter_coreset(X: np.ndarray, m, device: torch.device):
    N = X.shape[0]
    if m == "ALL" or (isinstance(m, int) and m >= N):
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(1234)
    sel = [int(rng.integers(0, N))]
    Xt = torch.from_numpy(X).to(device)
    centers = Xt[sel[-1]:sel[-1]+1]
    dmin = torch.cdist(Xt, centers).squeeze(1)
    for _ in tqdm(range(1, m), desc="coreset imgs", leave=False):
        idx = int(torch.argmax(dmin).item()); sel.append(idx)
        c = Xt[idx:idx+1]
        dmin = torch.minimum(dmin, torch.cdist(Xt, c).squeeze(1))
    return np.array(sel, dtype=np.int64)

# ------------------ TILE GRID ------------------
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
            h = max(1, y1 - y0); w = max(1, x1 - x0)
            rects.append((int(y0), int(x0), int(h), int(w)))
    return rects

class TileViewDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, rect, out_size=224):
        self.base = base_ds
        self.y, self.x, self.h, self.w = rect
        self.out = out_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        img_t, lbl, mask_t = self.base[i]   # img_t: (C,H,W) float[0,1]
        C, H, W = img_t.shape
        y, x, h, w = self.y, self.x, self.h, self.w
        y = max(0, min(y, H-1)); x = max(0, min(x, W-1))
        h = max(1, min(h, H-y)); w = max(1, min(w, W-x))
        img_tile = img_t[:, y:y+h, x:x+w]
        img_tile = F.interpolate(img_tile.unsqueeze(0), size=(self.out, self.out),
                                 mode='bilinear', align_corners=False).squeeze(0)
        img_tile = (img_tile - self.mean) / self.std
        if mask_t is not None:
            m = mask_t[y:y+h, x:x+w].float().unsqueeze(0).unsqueeze(0)
            m = F.interpolate(m, size=(self.out, self.out), mode="nearest").squeeze().cpu().numpy().astype(np.float32)
        else:
            m = None
        return img_tile, int(lbl), m

# ================== CHANNELS ==================
@dataclass
class ChannelMeta:
    h: int; w: int; span: float; spread: float

def compute_spread(stack: torch.Tensor) -> float:
    mu = stack.mean(dim=0)
    dif = stack - mu
    cov = (dif.t() @ dif) / max(1, stack.shape[0]-1)
    return float((torch.trace(cov) / cov.shape[0]).item())

class LazyBank:
    """Rappresenta un bank come lista di file .npy su disco (float32, shape (n_i, C))."""
    def __init__(self, paths: List[str], C: int):
        self.paths = paths
        self.C = C
        self._rows = None

    @property
    def rows(self) -> int:
        if self._rows is None:
            tot = 0
            for p in self.paths:
                with np.load(p, mmap_mode='r') as z:
                    tot += z['b'].shape[0]
            self._rows = int(tot)
        return self._rows

    def add_to_faiss(self, index, batch: int = 1_000_000):
        # Aggiunge a blocchi all'indice FAISS (evita un array gigante)
        for p in self.paths:
            with np.load(p, mmap_mode='r') as z:
                arr = z['b']   # float32 (n_i, C)
                n = arr.shape[0]
                for s in range(0, n, batch):
                    index.add(arr[s:s+batch])

    def iter_torch_batches(self, tile_b: int, device: torch.device):
        # Generator che produce tensori torch (tb, C) sul device
        for p in self.paths:
            with np.load(p, mmap_mode='r') as z:
                arr = z['b']   # (n_i, C) float32
                n = arr.shape[0]
                for s in range(0, n, tile_b):
                    yield torch.from_numpy(arr[s:s+tile_b]).to(device, non_blocking=True)

@torch.no_grad()
def build_channels_spill(
    F_train: torch.Tensor,              # (N,C,H,W) L2-normalized, on device
    img_anchors: np.ndarray,
    spill_dir: str,
    search_rad: int = 1,
    stride_h: int = 1, stride_w: int = 1,
    sim_min: float = 0.0,
    span_min: float = 0.0,
    spread_max: float = float("inf"),
    per_channel_limit: Optional[int] = None,
    batch_j: int = BATCH_J,
) -> Tuple[List[ChannelMeta], LazyBank]:

    if os.path.isdir(spill_dir):
        shutil.rmtree(spill_dir)
    os.makedirs(spill_dir, exist_ok=True)

    if isinstance(img_anchors, np.ndarray):
        img_anchors = torch.from_numpy(img_anchors).long()

    device = F_train.device
    N, C, H, W = F_train.shape
    ksize = 2*search_rad + 1; pad = search_rad

    hs = torch.arange(0, H, device=device, dtype=torch.long)[::stride_h]
    ws = torch.arange(0, W, device=device, dtype=torch.long)[::stride_w]
    Hs, Ws = torch.meshgrid(hs, ws, indexing="ij")
    coords = torch.stack([Hs.reshape(-1), Ws.reshape(-1)], 1)     # (L,2)
    L = coords.shape[0]

    lin_ids_full = (torch.arange(H, device=device).unsqueeze(1) * W +
                    torch.arange(W, device=device))               # (H,W)
    l_ids = lin_ids_full[hs][:, ws].reshape(-1)                   # (L,)

    use_amp = USE_AMP and (device.type == "cuda")
    chans: List[ChannelMeta] = []
    spill_paths: List[str] = []

    chunk_idx = 0
    for ia in tqdm(img_anchors.tolist(), desc="build channels (spill)", leave=False):
        A = F_train[ia]  # (C,H,W)
        A_vecs = A.permute(1,2,0)[hs][:, ws, :].reshape(-1, C).contiguous()  # (L,C)

        per_pos_acc: List[List[torch.Tensor]] = [[] for _ in range(L)]

        for j0 in range(0, N, batch_j):
            j1 = min(j0 + batch_j, N)
            Y = F_train[j0:j1]  # (B,C,H,W)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                patches = F.unfold(Y, kernel_size=ksize, padding=pad, stride=1)    # (B, C*K, H*W)
                patches_sel = patches[:, :, l_ids]                                 # (B, C*K, L)
                patches_sel = patches_sel.view(Y.size(0), C, ksize*ksize, L)       # (B, C, K, L)
                PS = patches_sel.permute(0, 3, 1, 2).contiguous()                  # (B, L, C, K)
                sims = torch.einsum('lc,blck->blk', A_vecs.to(PS.dtype), PS)       # (B, L, K)
                vals, argk = sims.max(dim=2)                                       # (B, L)

            valid_mask = (vals >= sim_min)
            idx = argk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, 1)
            best_vecs = torch.gather(PS, dim=3, index=idx).squeeze(-1)             # (B, L, C)

            for b in range(j1 - j0):
                vm = valid_mask[b]
                if vm.any():
                    l_valid = torch.nonzero(vm, as_tuple=False).squeeze(1).tolist()
                    bv = best_vecs[b].float().cpu()                                 # (L,C)
                    for l_id in l_valid:
                        per_pos_acc[l_id].append(bv[l_id:l_id+1])                   # (1,C)

            del Y, patches, patches_sel, PS, sims, vals, argk, valid_mask, idx, best_vecs
            if device.type == "cuda": torch.cuda.empty_cache()

        # per posizione -> valuta e salva su disco
        for l_id, acc_list in enumerate(per_pos_acc):
            if not acc_list: continue
            span = len(acc_list) / float(N)
            if span < span_min: continue

            stack = torch.cat(acc_list, dim=0)  # (valid, C) CPU float32
            spr = compute_spread(stack)
            if spr > spread_max: continue

            if per_channel_limit is not None and stack.shape[0] > per_channel_limit:
                sel = torch.randperm(stack.shape[0])[:per_channel_limit]
                stack = stack[sel]

            # SPILL: salva mini-chunk su disco come .npz (chiave 'b')
            path = os.path.join(spill_dir, f"bank_chunk_{chunk_idx:06d}.npz")
            np.savez_compressed(path, b=stack.numpy().astype(np.float32))
            spill_paths.append(path)
            chunk_idx += 1

            h, w = int(coords[l_id, 0].item()), int(coords[l_id, 1].item())
            chans.append(ChannelMeta(h=h, w=w, span=float(span), spread=float(spr)))

        del A, A_vecs, per_pos_acc
        if device.type == "cuda": torch.cuda.empty_cache()

    lazy_bank = LazyBank(spill_paths, C=C)
    return chans, lazy_bank

# ================== SCORING (streaming) ==================
@torch.no_grad()
def score_image_nn_faiss(
    F_img: torch.Tensor,                    # (C,Hf,Wf)
    bank: Union[torch.Tensor, LazyBank],    # tensor oppure LazyBank
    tile_q: int = TILE_Q
) -> torch.Tensor:
    C, Hf, Wf = F_img.shape
    Q = F_img.permute(1,2,0).reshape(-1, C).contiguous().cpu().numpy().astype(np.float32)  # (L,C)

    cpu_index = faiss.IndexFlatL2(C)
    use_gpu = faiss.get_num_gpus() > 0
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index

    # add bank
    if isinstance(bank, LazyBank):
        bank.add_to_faiss(index)
    else:
        B = bank.contiguous().cpu().numpy().astype(np.float32)
        index.add(B)

    L = Q.shape[0]
    out = np.empty((L,), dtype=np.float32)
    for q0 in range(0, L, tile_q):
        q1 = min(q0 + tile_q, L)
        D, _ = index.search(Q[q0:q1], k=1)
        out[q0:q1] = D[:, 0]

    if use_gpu:
        faiss.index_gpu_to_cpu(index)

    return torch.from_numpy(out.reshape(Hf, Wf))  # CPU

@torch.no_grad()
def score_image_nn_torch(
    F_img: torch.Tensor,                    # (C,Hf,Wf)
    bank: Union[torch.Tensor, LazyBank],    # tensor oppure LazyBank
    tile_q: int = 4096,
    tile_b: int = TILE_B,
    device: torch.device = torch.device(DEVICE)
) -> torch.Tensor:
    C, Hf, Wf = F_img.shape
    Q = F_img.permute(1,2,0).reshape(-1, C).to(device)  # (L,C)
    out = torch.full((Q.shape[0],), float("inf"), dtype=torch.float32, device=device)

    def bank_iter():
        if isinstance(bank, LazyBank):
            yield from bank.iter_torch_batches(tile_b, device)
        else:
            Bcpu = bank
            for b0 in range(0, Bcpu.shape[0], tile_b):
                yield Bcpu[b0:b0+tile_b].to(device, non_blocking=True)

    for q0 in range(0, Q.shape[0], tile_q):
        q1 = min(q0 + tile_q, Q.shape[0])
        q_blk = Q[q0:q1]  # (lq, C)
        best = torch.full((q1-q0,), float("inf"), dtype=torch.float32, device=device)

        for Bt in bank_iter():
            d = torch.cdist(q_blk, Bt)             # (lq, tb)
            m,_ = d.min(dim=1)
            best = torch.minimum(best, m)
            del Bt, d, m
            if device.type == "cuda": torch.cuda.empty_cache()

        out[q0:q1] = best

    return out.view(Hf, Wf).cpu()

# ================== PIPE TILED (per 1 tile) ==================
def inreach_for_one_tile(train_tile_ds, val_tile_ds, device, tile_tag: str):
    pin = (device.type == "cuda")
    tr_loader = torch.utils.data.DataLoader(train_tile_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=pin)
    va_loader = torch.utils.data.DataLoader(val_tile_ds,   batch_size=16, shuffle=False, num_workers=0, pin_memory=pin)

    model, outs = get_backbone(device)

    # features train
    F_tr_raw, G_tr = extract_features(model, outs, tr_loader, device)
    F_tr = l2norm(F_tr_raw, dim=1).to(device)

    anchors = kcenter_coreset(G_tr.numpy().astype(np.float32), m=CORESET_IMGS, device=device)

    # build channels + bank (SPILL)
    spill_dir = os.path.join(SPILL_ROOT, f"{CODICE_PEZZO}_{tile_tag}_{uuid.uuid4().hex[:8]}")
    _, lazy_bank = build_channels_spill(
        F_train=F_tr,
        img_anchors=anchors,
        spill_dir=spill_dir,
        search_rad=SEARCH_RAD,
        stride_h=STRIDE_H, stride_w=STRIDE_W,
        sim_min=SIM_MIN,
        span_min=SPAN_MIN,
        spread_max=SPREAD_MAX,
        per_channel_limit=BANK_PER_CHANNEL_LIMIT,
        batch_j=BATCH_J,
    )
    if lazy_bank.rows == 0:
        raise RuntimeError("Bank vuoto per il tile: allenta i filtri o cambia STRIDE/SEARCH_RAD.")

    # features val
    F_val_raw, _ = extract_features(model, outs, va_loader, device)
    F_val = l2norm(F_val_raw, dim=1).to(device)

    # inference
    raw_maps, img_scores = [], []
    idx_feat = 0
    with torch.inference_mode():
        for (x, y, _) in va_loader:
            Bsz = y.shape[0]
            f_batch = F_val[idx_feat:idx_feat+Bsz]; idx_feat += Bsz
            for b in range(Bsz):
                f = f_batch[b]  # (C,Hf,Wf)
                if USE_FAISS and _FAISS_OK:
                    dist_map = score_image_nn_faiss(f, lazy_bank, tile_q=TILE_Q)
                else:
                    dist_map = score_image_nn_torch(f, lazy_bank, tile_q=4096, tile_b=TILE_B, device=device)
                M = dist_map.unsqueeze(0).unsqueeze(0)
                mup = F.interpolate(M, size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
                mup = gaussian_filter(mup, sigma=GAUSS_SIGMA).astype(np.float32)
                raw_maps.append(mup); img_scores.append(float(mup.max()))

    # pulizia
    del tr_loader, va_loader, F_tr, F_val, F_tr_raw, F_val_raw, model, outs
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    return raw_maps, np.array(img_scores, dtype=np.float32), spill_dir  # ritorno dir per eventuale debug

# ================== MAIN TILED ==================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device(DEVICE)
    _p("**device:", device)

    # dataset full-res (niente resize qui; il tile fa il crop → 224)
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO, img_size=None,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=SEED, transform=None, rgb_policy="fullres_only"
    )
    _p("[meta]", meta)

    # misura da una immagine
    sample_img, _, _ = train_set[0]
    _, H0, W0 = sample_img.shape

    grid = compute_tile_grid_by_counts(H0, W0, TILED_ROWS, TILED_COLS, overlap=TILED_OVERLAP)
    _p(f"[grid] {len(grid)} tile -> {grid[:4]}")

    all_maps: List[List[np.ndarray]] = []   # lista per-tile di heatmap (ordine val_set)
    all_scores: List[np.ndarray] = []

    # fai girare ogni tile in sequenza (minima RAM)
    for t_id, rect in enumerate(grid):
        _p(f"\n=== TILE {t_id+1}/{len(grid)} rect={rect} ===")
        train_tile_ds = TileViewDataset(train_set, rect, out_size=IMG_SIZE)
        val_tile_ds   = TileViewDataset(val_set,   rect, out_size=IMG_SIZE)

        tile_tag = f"tile{t_id}_r{TILED_ROWS}c{TILED_COLS}"
        maps_t, scores_t, spill_dir = inreach_for_one_tile(train_tile_ds, val_tile_ds, device, tile_tag)
        all_maps.append(maps_t)
        all_scores.append(scores_t)

        # opzionale: puoi cancellare lo spill per liberare spazio
        try: shutil.rmtree(spill_dir)
        except Exception: pass

    # ricomposizione semplice (media dei tile upscalati a full-res)
    # NB: per restare essenziale/robusto, qui ricollochiamo ogni heatmap al suo rettangolo senza smoothing finestra.
    num_val = len(val_set)
    acc = [np.zeros((H0, W0), np.float32) for _ in range(num_val)]
    wgt = [np.zeros((H0, W0), np.float32) for _ in range(num_val)]
    for t_id, rect in enumerate(grid):
        y, x, h, w = rect
        for i in range(num_val):
            hmap = all_maps[t_id][i]
            tile_hw = F.interpolate(torch.from_numpy(hmap)[None,None], size=(h, w),
                                    mode="bilinear", align_corners=False).squeeze().numpy().astype(np.float32)
            acc[i][y:y+h, x:x+w] += tile_hw
            wgt[i][y:y+h, x:x+w] += 1.0

    full_res_maps = [(acc[i] / np.maximum(wgt[i], 1e-6)).astype(np.float32) for i in range(num_val)]
    agg_scores = np.mean(np.stack(all_scores, 0), axis=0).astype(np.float32)  # media sugli score tile

    # valutazione (PRO + AUROC)
    results = run_pixel_level_evaluation(
        score_map_list=full_res_maps,
        val_set=val_set,
        img_scores=agg_scores,
        use_threshold="pro",
        fpr_limit=0.01,
        vis=True,
        vis_ds_or_loader=val_set
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO} | tiled {TILED_ROWS}x{TILED_COLS}")

if __name__ == "__main__":
    main()
