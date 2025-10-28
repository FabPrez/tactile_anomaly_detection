# InReach_fast.py — InReaCh ufficiale ottimizzato (build channels batched+AMP, FAISS GPU con fallback, I/O opzionale)

import os, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# ------------------------------------------------------------
# Switch di I/O (cache)
# ------------------------------------------------------------
PERSIST_FEATURES  = True   # salva/carica F_tr, F_val
PERSIST_BANK      = False   # salva/carica bank npz
PERSIST_CHANNELS  = False  # salva/carica channels npz (disattivato per velocità)

torch.backends.cudnn.benchmark = True

# ===== FAISS availability =====
USE_FAISS = True
_FAISS_OK = False
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

# ===== config =====
METHOD        = "INREACH_OFFICIAL"
CODICE_PEZZO  = "PZ3"

TRAIN_POSITIONS   = ["pos2"]
VAL_GOOD_PER_POS  = 0
VAL_GOOD_SCOPE    = ["pos2"]
VAL_FAULT_SCOPE   = ["pos2"]
GOOD_FRACTION     = 1.0

IMG_SIZE    = 224
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Backbone/layer
FEATURE_LAYER = "layer2"          # hook a layer2[-1]

# Coreset immagini (ancore). "ALL" = tutte (come paper/repo)
CORESET_IMGS   = "ALL"

# Matching locale per canali
SEARCH_RAD     = 1                # 3x3
SIM_MIN        = 0.0
STRIDE_H       = 1                # puoi alzare a 2 per ridurre 4x il bank
STRIDE_W       = 1
# Filtri canale
SPAN_MIN       = 0.0
SPREAD_MAX     = float("inf")

# Limite opzionale patch per canale (RAM). None = no limit
BANK_PER_CHANNEL_LIMIT = None     # es. 128 per ridurre RAM/VRAM

# Costruzione canali (performance)
BATCH_J        = 48               # immagini train per blocco durante build channels
USE_AMP        = True             # mixed precision su GPU

# Inference NN L2
TILE_Q         = 65536            # query per batch (FAISS)
TILE_B         = 20000            # chunk bank per torch.cdist
GAUSS_SIGMA    = 2.0

# Logging
def _p(*a, **k):
    print(*a, **k)

# ================== UTIL BASE ==================
def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def get_backbone(device):
    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
    m.eval()
    outs = []
    def hook(_m,_i,o): outs.append(o)
    m.layer2[-1].register_forward_hook(hook)  # feature intermedie
    m.avgpool.register_forward_hook(hook)     # globali per (eventuale) coreset
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
    # m == "ALL" -> tutti gli indici (purista)
    N = X.shape[0]
    if m == "ALL" or (isinstance(m, int) and m >= N):
        return np.arange(N, dtype=np.int64)
    # Greedy k-center (se vuoi campionare davvero)
    rng = np.random.default_rng(1234)
    sel = [int(rng.integers(0, N))]
    Xt = torch.from_numpy(X).to(device)
    centers = Xt[sel[-1]:sel[-1]+1]
    dmin = torch.cdist(Xt, centers).squeeze(1)
    for _ in tqdm(range(1, m), desc="coreset imgs", leave=False):
        idx = int(torch.argmax(dmin).item())
        sel.append(idx)
        c = Xt[idx:idx+1]
        dmin = torch.minimum(dmin, torch.cdist(Xt, c).squeeze(1))
    return np.array(sel, dtype=np.int64)

# ========= Utility VRAM per FAISS GPU fallback =========
def _free_vram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    free_bytes, _ = torch.cuda.mem_get_info()
    return int(free_bytes)

def _should_use_faiss_gpu_for_bank(bank_cpu: torch.Tensor, safety: float = 0.6) -> bool:
    """
    Ritorna True se il bank (float32) entra in VRAM con un margine di sicurezza.
    Stima memoria: M*C*4 byte. Se non c'è GPU/FAISS, restituisce False.
    """
    if not (_FAISS_OK and torch.cuda.is_available()):
        return False
    need = bank_cpu.shape[0] * bank_cpu.shape[1] * 4  # float32
    return need < int(safety * _free_vram_bytes())

# ================== CHANNELS ==================
@dataclass
class ChannelMeta:
    h: int
    w: int
    span: float
    spread: float

def compute_spread(stack: torch.Tensor) -> float:
    mu = stack.mean(dim=0)
    dif = stack - mu
    cov = (dif.t() @ dif) / max(1, stack.shape[0]-1)
    return float((torch.trace(cov) / cov.shape[0]).item())

@torch.no_grad()
def build_channels(
    F_train: torch.Tensor,              # (N,C,H,W) L2-normalized on channels, on device
    img_anchors: np.ndarray,
    search_rad: int = 1,
    stride_h: int = 1, stride_w: int = 1,
    sim_min: float = 0.0,
    span_min: float = 0.0,
    spread_max: float = float("inf"),
    per_channel_limit: Optional[int] = None,
    batch_j: int = BATCH_J,
) -> Tuple[List[ChannelMeta], torch.Tensor]:

    if isinstance(img_anchors, np.ndarray):
        img_anchors = torch.from_numpy(img_anchors).long()

    device = F_train.device
    N, C, H, W = F_train.shape
    ksize = 2*search_rad + 1
    pad = search_rad

    # griglia con stride
    hs = torch.arange(0, H, device=device, dtype=torch.long)[::stride_h]
    ws = torch.arange(0, W, device=device, dtype=torch.long)[::stride_w]
    Hs, Ws = torch.meshgrid(hs, ws, indexing="ij")
    coords = torch.stack([Hs.reshape(-1), Ws.reshape(-1)], 1)     # (L,2)
    L = coords.shape[0]

    # indici lineari (stride=1) -> sotto-campionati via (hs,ws)
    lin_ids_full = (torch.arange(H, device=device).unsqueeze(1) * W +
                    torch.arange(W, device=device))               # (H,W)
    l_ids = lin_ids_full[hs][:, ws].reshape(-1)                   # (L,)

    chans: List[ChannelMeta] = []
    bank_chunks: List[torch.Tensor] = []
    use_amp = USE_AMP and (device.type == "cuda")

    for ia in tqdm(img_anchors.tolist(), desc="build channels", leave=False):
        A = F_train[ia]  # (C,H,W)
        # vettori ancora alle posizioni (stride)
        A_vecs = A.permute(1,2,0)[hs][:, ws, :].reshape(-1, C).contiguous()  # (L,C)

        per_pos_acc = [[] for _ in range(L)]

        for j0 in range(0, N, batch_j):
            j1 = min(j0 + batch_j, N)
            Y = F_train[j0:j1]  # (B,C,H,W)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                patches = F.unfold(Y, kernel_size=ksize, padding=pad, stride=1)    # (B, C*K, H*W)
                patches_sel = patches[:, :, l_ids]                                 # (B, C*K, L)
                patches_sel = patches_sel.view(Y.size(0), C, ksize*ksize, L)       # (B, C, K, L)
                PS = patches_sel.permute(0, 3, 1, 2).contiguous()                  # (B, L, C, K)

                # dot per ogni (b,l,k): <A_vecs[l], PS[b,l,:,k]> -> (B,L,K)
                sims = torch.einsum('lc,blck->blk', A_vecs.to(PS.dtype), PS)       # (B, L, K)
                vals, argk = sims.max(dim=2)                                       # (B, L)

            valid_mask = (vals >= sim_min)                                         # (B, L)

            # gather vettore migliore mantenendo K=1
            idx = argk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, 1)            # (B, L, C, 1)
            best_vecs = torch.gather(PS, dim=3, index=idx).squeeze(-1)             # (B, L, C)

            # Accumulo CPU (float32) per ciascuna posizione l
            for b in range(j1 - j0):
                vm = valid_mask[b]
                if vm.any():
                    l_valid = torch.nonzero(vm, as_tuple=False).squeeze(1).tolist()
                    bv = best_vecs[b].float().cpu()                                 # (L,C)
                    for l_id in l_valid:
                        per_pos_acc[l_id].append(bv[l_id:l_id+1])                   # (1,C)

            # cleanup
            del Y, patches, patches_sel, PS, sims, vals, argk, valid_mask, idx, best_vecs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Valutazione/filtro canale
        for l_id, acc_list in enumerate(per_pos_acc):
            if not acc_list:
                continue
            span = len(acc_list) / float(N)
            if span < span_min:
                continue

            stack = torch.cat(acc_list, dim=0)          # (valid, C) CPU float32
            spr = compute_spread(stack)
            if spr > spread_max:
                continue

            if per_channel_limit is not None and stack.shape[0] > per_channel_limit:
                sel = torch.randperm(stack.shape[0])[:per_channel_limit]
                stack = stack[sel]

            bank_chunks.append(stack)
            h, w = int(coords[l_id, 0].item()), int(coords[l_id, 1].item())
            chans.append(ChannelMeta(h=h, w=w, span=float(span), spread=float(spr)))

        del A, A_vecs, per_pos_acc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bank = torch.cat(bank_chunks, dim=0).contiguous() if bank_chunks else torch.empty((0, C), dtype=torch.float32)
    return chans, bank

# ================== NOMINAL MODEL I/O ==================
def _features_dir_for(part: str, train_tag: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "Dataset", part, "features", train_tag)
    os.makedirs(base, exist_ok=True)
    return base

def save_channels_npz(chans: List[ChannelMeta], part: str, train_tag: str, method: str):
    if not PERSIST_CHANNELS:
        return None
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_channels_train.npz")
    if len(chans) == 0:
        np.savez_compressed(path, h=np.zeros((0,), np.int16), w=np.zeros((0,), np.int16),
                            span=np.zeros((0,), np.float16), spread=np.zeros((0,), np.float16))
    else:
        h = np.array([c.h for c in chans], dtype=np.int16)
        w = np.array([c.w for c in chans], dtype=np.int16)
        span = np.array([c.span for c in chans], dtype=np.float16)
        spread = np.array([c.spread for c in chans], dtype=np.float16)
        np.savez_compressed(path, h=h, w=w, span=span, spread=spread)
    return path

def load_channels_npz(part: str, train_tag: str, method: str) -> List[ChannelMeta]:
    if not PERSIST_CHANNELS:
        raise FileNotFoundError("channels cache disabled")
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_channels_train.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    h = data["h"].astype(np.int32); w = data["w"].astype(np.int32)
    span = data["span"].astype(np.float32); spread = data["spread"].astype(np.float32)
    chans = [ChannelMeta(h=int(h[i]), w=int(w[i]), span=float(span[i]), spread=float(spread[i])) for i in range(h.shape[0])]
    return chans

def save_bank_npz(bank: torch.Tensor, part: str, train_tag: str, method: str):
    if not PERSIST_BANK:
        return None
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_bank_train.npz")
    np.savez_compressed(path, bank=bank.numpy().astype(np.float16))
    return path

def load_bank_npz(part: str, train_tag: str, method: str) -> torch.Tensor:
    if not PERSIST_BANK:
        raise FileNotFoundError("bank cache disabled")
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_bank_train.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    B = torch.from_numpy(data["bank"].astype(np.float32))   # (M,C) CPU
    return B

# ================== SCORING (NN L2) ==================
@torch.no_grad()
def score_image_nn_faiss(
    F_img: torch.Tensor,      # (C,Hf,Wf) L2-normalized on channels (device-agnostic)
    bank_cpu: torch.Tensor,   # (M,C) CPU float32
    tile_q: int = TILE_Q
) -> torch.Tensor:
    C, Hf, Wf = F_img.shape
    Q = F_img.permute(1,2,0).reshape(-1, C).contiguous().cpu().numpy().astype(np.float32)

    # Evita copie superflue: se già float32 lascia stare
    if bank_cpu.dtype != torch.float32:
        bank_cpu = bank_cpu.float()
    B = bank_cpu.contiguous().cpu().numpy()  # niente .astype(np.float32) per non raddoppiare RAM

    cpu_index = faiss.IndexFlatL2(C)

    # Usa la GPU solo se il bank entra in VRAM (margine di sicurezza)
    use_gpu = _should_use_faiss_gpu_for_bank(bank_cpu)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index

    index.add(B)  # build

    L = Q.shape[0]
    out = np.empty((L,), dtype=np.float32)
    for q0 in range(0, L, tile_q):
        q1 = min(q0 + tile_q, L)
        D, _ = index.search(Q[q0:q1], k=1)  # D: (lq, 1)
        out[q0:q1] = D[:, 0]

    if use_gpu:
        faiss.index_gpu_to_cpu(index)

    return torch.from_numpy(out.reshape(Hf, Wf))  # CPU

@torch.no_grad()
def score_image_nn_torch(
    F_img: torch.Tensor,      # (C,Hf,Wf) L2-normalized on channels
    bank_cpu: torch.Tensor,   # (M,C) CPU float32
    tile_q: int = 4096,
    tile_b: int = TILE_B,
    device: torch.device = torch.device(DEVICE)
) -> torch.Tensor:
    C, Hf, Wf = F_img.shape
    Q = F_img.permute(1,2,0).reshape(-1, C).to(device)  # (L,C)
    out = torch.full((Q.shape[0],), float("inf"), dtype=torch.float32, device=device)

    for q0 in range(0, Q.shape[0], tile_q):
        q1 = min(q0 + tile_q, Q.shape[0])
        q_blk = Q[q0:q1]  # (lq, C)
        best = torch.full((q1-q0,), float("inf"), dtype=torch.float32, device=device)

        for b0 in range(0, bank_cpu.shape[0], tile_b):
            b1 = min(b0 + tile_b, bank_cpu.shape[0])
            Bt = bank_cpu[b0:b1].to(device, non_blocking=True)  # (tb, C)
            d = torch.cdist(q_blk, Bt)                          # (lq, tb)
            m,_ = d.min(dim=1)
            best = torch.minimum(best, m)
            del Bt, d, m
            if device.type == "cuda":
                torch.cuda.empty_cache()

        out[q0:q1] = best

    return out.view(Hf, Wf).cpu()

# ================== MAIN ==================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device(DEVICE)

    # ===== DATA =====
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO, img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=SEED, transform=None
    )
    TRAIN_TAG = meta["train_tag"]
    _p(f"Train GOOD: {meta['counts']['train_good']} | Val TOT: {meta['counts']['val_total']}")

    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=16, device=device)

    # ===== MODEL =====
    model, outs = get_backbone(device)

    # ===== FEATURE (TRAIN) =====
    try:
        if not PERSIST_FEATURES:
            raise FileNotFoundError
        payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        F_tr = payload["F_tr"]; G_tr = payload["G_tr"]
    except FileNotFoundError:
        F_tr_raw, G_tr = extract_features(model, outs, train_loader, device)
        F_tr = l2norm(F_tr_raw, dim=1).cpu()
        if PERSIST_FEATURES:
            save_split_pickle({"F_tr":F_tr, "G_tr":G_tr}, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    # ===== IMAGE CORESEt =====
    anchors = kcenter_coreset(G_tr.numpy().astype(np.float32), m=CORESET_IMGS, device=device)

    # ===== BUILD CHANNELS + BANK =====
    bank_loaded = False
    try:
        bank_cpu = load_bank_npz(CODICE_PEZZO, TRAIN_TAG, METHOD)
        bank_loaded = True
    except FileNotFoundError:
        pass

    if not bank_loaded:
        F_tr_dev = F_tr.to(device)
        chans, bank_cpu = build_channels(
            F_train=F_tr_dev,
            img_anchors=anchors,
            search_rad=SEARCH_RAD,
            stride_h=STRIDE_H, stride_w=STRIDE_W,
            sim_min=SIM_MIN,
            span_min=SPAN_MIN,
            spread_max=SPREAD_MAX,
            per_channel_limit=BANK_PER_CHANNEL_LIMIT,
            batch_j=BATCH_J,
        )
        # opzionale: salva
        save_channels_npz(chans, CODICE_PEZZO, TRAIN_TAG, METHOD)
        save_bank_npz(bank_cpu, CODICE_PEZZO, TRAIN_TAG, METHOD)

    if bank_cpu.shape[0] == 0:
        raise RuntimeError("Nominal model (bank) vuoto: allenta filtri o controlla SEARCH_RAD/STRIDE.")

    # ===== FEATURE (VAL) =====
    try:
        if not PERSIST_FEATURES:
            raise FileNotFoundError
        val_pack = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)
        F_val_raw, gt_list = val_pack["F_val_raw"], val_pack["labels"]
    except FileNotFoundError:
        F_val_raw, _ = extract_features(model, outs, val_loader, device)  # (Nv,C,Hf,Wf)
        gt_list = []
        for _, y, _ in val_loader: gt_list.extend(y.cpu().numpy())
        if PERSIST_FEATURES:
            save_split_pickle({"F_val_raw":F_val_raw, "labels": np.array(gt_list, dtype=np.int64)},
                              CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)

    F_val = l2norm(F_val_raw, dim=1).to(device)
    _, C, Hf, Wf = F_val.shape

    # ===== INFERENCE =====
    raw_maps, img_scores, gt_list_out = [], [], []
    with torch.inference_mode():
        total_imgs, idx_feat = F_val.shape[0], 0
        pbar = tqdm(total=total_imgs, desc="InReaCh inference", leave=False)

        for (x, y, _) in val_loader:
            Bsz = y.shape[0]
            f_batch = F_val[idx_feat:idx_feat+Bsz]; idx_feat += Bsz
            for b in range(Bsz):
                gt_list_out.append(int(y[b].item()))
                f = f_batch[b]  # (C,Hf,Wf)

                if USE_FAISS and _FAISS_OK:
                    dist_map = score_image_nn_faiss(f, bank_cpu, tile_q=TILE_Q)
                else:
                    dist_map = score_image_nn_torch(f, bank_cpu, tile_q=4096, tile_b=TILE_B, device=device)

                M = dist_map.unsqueeze(0).unsqueeze(0)
                mup = F.interpolate(M, size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
                mup = gaussian_filter(mup, sigma=GAUSS_SIGMA).astype(np.float32)
                raw_maps.append(mup); img_scores.append(float(mup.max()))
                pbar.update(1)

        pbar.close()

    img_scores = np.array(img_scores, dtype=np.float32)
    gt_np = np.asarray(gt_list_out, dtype=np.int32)

    # ----- image-level AUROC -----
    fpr, tpr, thr = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    J = tpr - fpr
    best_idx = int(np.argmax(J)); best_thr = float(thr[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0,1]).ravel()
    _p(f"[image-level] AUC={auc_img:.3f}  thr(Youden)={best_thr:.6f}  TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    # ----- pixel-level -----
    results = run_pixel_level_evaluation(
        score_map_list=raw_maps,
        val_set=val_set,
        img_scores=img_scores,
        use_threshold="pro",
        fpr_limit=0.01,
        vis=True,
        vis_ds_or_loader=val_loader.dataset
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")

if __name__ == "__main__":
    main()
