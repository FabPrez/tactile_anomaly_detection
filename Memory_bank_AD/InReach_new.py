# inreach_official.py
# Goal: mirror the official InReaCh repo/paper behavior as closely as possible
# Key choices for "max fidelity" (this version):
#  - WRN50-2, features from layer1 (default)
#  - ImageNet normalization inside model (equivalente a Normalize esterno)
#  - Seeds D = primi D (nessuno shuffle)  [SHUFFLE_SEEDS = False]
#  - Channel building: mutual-nearest verso il centro del seme, un patch/immagine/canale, assegnazione unica
#  - Niente positional embedding o rotazioni
#  - Nessun local averaging (AGG_K = 1)
#  - 1-NN L2 esatto (FAISS se disponibile, altrimenti torch.cdist)

import os, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

torch.backends.cudnn.benchmark = True

# ---- FAISS (optional) ----
USE_FAISS = True
_FAISS_OK = False
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

# ---- local deps ----
from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# ================= CONFIG (faithful defaults) =================
METHOD = "INREACH_OFFICIAL"
CODICE_PEZZO = "PZ2"

TRAIN_POSITIONS   = ["pos5"]
VAL_GOOD_PER_POS  = 0
VAL_GOOD_SCOPE    = ["pos5"]
VAL_FAULT_SCOPE   = ["pos5"]
GOOD_FRACTION     = 1.0

IMG_SIZE    = 224
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Feature pipeline (faithful): shallow features only
FEATURE_LAYER      = "layer1"   # official favors shallow map ("layer1"); "layer2" opzionale
AGG_K              = 3          # <<< MOD: disattivato local averaging per massima fedeltà
USE_POS_EMB        = False      # <<< MOD: niente positional embedding
POS_EMB_WEIGHT     = 0.12       # ignorato se USE_POS_EMB=False
POS_TEST_TAU       = 0.990      # ignorato se USE_POS_EMB=False
TRY_ROT_ALIGN      = False      # <<< MOD: niente hard rotation alignment
NORMALIZE_IMAGENET = True       # normalizzazione ImageNet dentro al modello

# Channel building (faithful)
ASSOC_DEPTH_D     = 10           # ~10 suggerito nel paper; qui 8 per RAM
SEARCH_RAD        = 1           # K = 2R+1 neighborhood (R=1 → 3x3)
SIM_MIN           = 0.0         # keep 0 unless needed
MIN_CHANNEL_SPAN  = 0.05        # fraction di realizzazioni nel canale ≥ 5%
MAX_CHANNEL_STD   = 0.60        # mean std across dims ≤ 0.60 (tunable)
STRIDE_H          = 1           # stride 1 per fedeltà
STRIDE_W          = 1
SHUFFLE_SEEDS     = True       # <<< MOD: semi = primi D senza shuffle

# Inference
TILE_Q         = 65536
TILE_B         = 20000
GAUSS_SIGMA    = 2.0

# I/O caches
SAVE_CHANNELS  = False
LOAD_CHANNELS  = False

# ==================== UTILS ====================

def set_seeds(seed:int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class ShallowWRN(nn.Module):
    """WRN-50-2 con hook su un singolo layer e avgpool, + (opzionale) normalizzazione ImageNet."""
    def __init__(self, layer_name: str, device: torch.device):
        super().__init__()
        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
        self.model.eval()
        self.layer_name = layer_name
        self._buf: Dict[str, torch.Tensor] = {}

        def hook_factory(name):
            def _h(m,i,o):
                self._buf[name] = o
            return _h

        if layer_name == "layer1":
            self.model.layer1[-1].register_forward_hook(hook_factory("feat"))
        elif layer_name == "layer2":
            self.model.layer2[-1].register_forward_hook(hook_factory("feat"))
        else:
            raise ValueError("FEATURE_LAYER must be 'layer1' or 'layer2'")

        self.model.avgpool.register_forward_hook(hook_factory("avg"))

        # --- buffer mean/std per normalizzazione ImageNet ---
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x atteso in [0,1] dal data_loader; normalizziamo come in repo ufficiale
        if NORMALIZE_IMAGENET:
            x = x.to(dtype=torch.float32)
            x = (x - self.imagenet_mean) / self.imagenet_std

        _ = self.model(x)
        f = self._buf["feat"]  # (B,C,H,W)
        if AGG_K and AGG_K > 1:
            pad = AGG_K // 2
            f = F.avg_pool2d(f, kernel_size=AGG_K, stride=1, padding=pad)
        g = torch.flatten(self._buf["avg"], 1)  # (B,2048)
        return f, g

# ---- positional embedding (DISABLED di default) ----
@torch.no_grad()
def make_positional_embedding(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, steps=h, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    pe = torch.stack([Y, X], dim=0)  # (2, H, W)
    return pe

# ================= CHANNELS (Algorithm-1 style) =================
@dataclass
class ChannelMeta:
    h: int
    w: int
    span: float
    mean_std: float

@torch.no_grad()
def build_channels_official(
    F_train: torch.Tensor,              # (N,C,H,W) L2-normalized per channel, on DEVICE
    assoc_depth: int = 8,
    search_rad: int = 1,
    stride_h: int = 1, stride_w: int = 1,
    sim_min: float = 0.0,
    min_span: float = 0.0,
    max_channel_std: float = 1.0,
    shuffle_seeds: bool = False,        # default coerente con SHUFFLE_SEEDS di file
) -> Tuple[List[ChannelMeta], torch.Tensor]:
    device = F_train.device
    N, C, H, W = F_train.shape
    ksize = 2*search_rad + 1
    pad = search_rad

    # grid con stride = 1
    hs = torch.arange(0, H, device=device, dtype=torch.long)[::stride_h]
    ws = torch.arange(0, W, device=device, dtype=torch.long)[::stride_w]
    Hs, Ws = torch.meshgrid(hs, ws, indexing="ij")
    coords = torch.stack([Hs.reshape(-1), Ws.reshape(-1)], 1)  # (L,2)
    L = coords.shape[0]

    lin_ids_full = (torch.arange(H, device=device).unsqueeze(1) * W + torch.arange(W, device=device))
    l_ids = lin_ids_full[hs][:, ws].reshape(-1)

    # maschera per impedire multi-assegnazione della stessa patch (per immagine)
    used_patch_mask = [torch.zeros((H*W,), dtype=torch.bool, device=device) for _ in range(N)]
    center_k = search_rad * ksize + search_rad  # indice del centro nel vicinato

    # --- scelta seeds: primi D (senza shuffle) ---
    idx_all = list(range(N))
    if shuffle_seeds:
        rng = random.Random(SEED)
        rng.shuffle(idx_all)
    seed_idx = idx_all[:min(max(1, assoc_depth), N)]

    # --- immagini non-seme (<<< MOD: escludiamo i semi dal loop) ---
    non_seed = [j for j in range(N) if j not in seed_idx]

    chans: List[ChannelMeta] = []
    bank_chunks: List[torch.Tensor] = []

    for s in tqdm(seed_idx, desc="| build channels (seeds) |", leave=False):
        S = F_train[s:s+1]
        patches_seed = F.unfold(S, kernel_size=ksize, padding=pad, stride=1)     # (1, C*K, H*W)
        patches_seed_sel = patches_seed[:, :, l_ids].view(1, C, ksize*ksize, L).squeeze(0)  # (C,K,L)
        A_vecs = S[0].permute(1,2,0)[hs][:, ws, :].reshape(-1, C).contiguous()   # (L,C)

        per_channel_acc: List[List[torch.Tensor]] = [[] for _ in range(L)]
        per_channel_img_added = [set() for _ in range(L)]

        # <<< MOD: scorriamo solo le immagini NON-SEME >>>
        for j in range(N):
            Y = F_train[j:j+1]
            patches_j = F.unfold(Y, kernel_size=ksize, padding=pad, stride=1)     # (1, C*K, H*W)
            patches_j_sel = patches_j[:, :, l_ids].view(1, C, ksize*ksize, L).squeeze(0)  # (C,K,L)

            # best-matching nel vicinato di ogni centro (A_vecs)
            sims_j = torch.einsum('lc,ckl->kl', A_vecs, patches_j_sel)  # (K,L)
            vals, argk = torch.max(sims_j, dim=0)                       # (L,)

            # vettori y* selezionati
            gather_idx = argk.view(1,1,-1).expand(C,1,L)
            ystar = torch.gather(patches_j_sel, dim=1, index=gather_idx).squeeze(1)  # (C,L)

            # mutual verso il seed: il centro del seed deve essere il nearest nel suo vicinato
            sims_seed = torch.einsum('cl,ckl->kl', ystar, patches_seed_sel)  # (K,L)
            argk_seed = torch.argmax(sims_seed, dim=0)                       # (L,)

            valid = (vals >= sim_min) & (argk_seed == center_k)
            if valid.any():
                l_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                for li in l_idx.tolist():
                    if j in per_channel_img_added[li]:
                        continue
                    k_sel = int(argk[li].item())
                    dh = (k_sel // ksize) - search_rad
                    dw = (k_sel %  ksize) - search_rad
                    # li indicizza la griglia (hs,ws) → back to (h_c, w_c)
                    h_c = int(coords[li, 0].item())
                    w_c = int(coords[li, 1].item())
                    h0 = h_c + dh
                    w0 = w_c + dw
                    if h0 < 0 or h0 >= H or w0 < 0 or w0 >= W:
                        continue
                    abs_lin = h0 * W + w0
                    if used_patch_mask[j][abs_lin]:
                        continue
                    used_patch_mask[j][abs_lin] = True
                    per_channel_img_added[li].add(j)
                    vec = patches_j_sel[:, k_sel, li].detach()  # (C,)
                    per_channel_acc[li].append(vec.unsqueeze(0))

            del Y, patches_j, patches_j_sel, sims_j, vals, argk, gather_idx, ystar, sims_seed, argk_seed, valid
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # trimming per posizione di canale
        for li in range(L):
            stack_list = per_channel_acc[li]
            if len(stack_list) == 0:
                continue
            stack = torch.cat(stack_list, dim=0)  # (m,C)
            span = len(per_channel_img_added[li]) / float(N)
            if span < min_span:
                continue
            # sigma ufficiale: media delle std per-dimensione (unbiased=False)
            mean_std = torch.mean(torch.std(stack, dim=0, unbiased=False)).item()
            if mean_std > max_channel_std:
                continue
            bank_chunks.append(stack.detach().cpu().float())
            h_c = int(coords[li,0].item()); w_c = int(coords[li,1].item())
            chans.append(ChannelMeta(h=h_c, w=w_c, span=float(span), mean_std=float(mean_std)))

        del S, patches_seed, patches_seed_sel, A_vecs, per_channel_acc, per_channel_img_added
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bank = torch.cat(bank_chunks, dim=0).contiguous() if len(bank_chunks)>0 else torch.empty((0, C), dtype=torch.float32)
    return chans, bank

# ================= NOMINAL MODEL I/O =================

def _features_dir_for(part: str, train_tag: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "Dataset", part, "features", train_tag)
    os.makedirs(base, exist_ok=True)
    return base

def save_channels_npz(chans: List[ChannelMeta], part: str, train_tag: str, method: str):
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_channels_train.npz")
    if len(chans) == 0:
        np.savez_compressed(path, h=np.zeros((0,), np.int16), w=np.zeros((0,), np.int16),
                            span=np.zeros((0,), np.float16), mean_std=np.zeros((0,), np.float16))
    else:
        h = np.array([c.h for c in chans], dtype=np.int16)
        w = np.array([c.w for c in chans], dtype=np.int16)
        span = np.array([c.span for c in chans], dtype=np.float16)
        ms  = np.array([c.mean_std for c in chans], dtype=np.float16)
        np.savez_compressed(path, h=h, w=w, span=span, mean_std=ms)
    print(f"[npz] saved channels: {path}")
    return path

def load_channels_npz(part: str, train_tag: str, method: str) -> List[ChannelMeta]:
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_channels_train.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    h = data["h"].astype(np.int32); w = data["w"].astype(np.int32)
    span = data["span"].astype(np.float32); ms = data["mean_std"].astype(np.float32)
    chans = [ChannelMeta(h=int(h[i]), w=int(w[i]), span=float(span[i]), mean_std=float(ms[i])) for i in range(h.shape[0])]
    print(f"[npz] loaded {len(chans)} channels from {path}")
    return chans

def save_bank_npz(bank: torch.Tensor, part: str, train_tag: str, method: str):
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_bank_train.npz")
    np.savez_compressed(path, bank=bank.numpy().astype(np.float16))
    print(f"[npz] saved bank: {path}  (vecs={bank.shape[0]}, dim={bank.shape[1]})")
    return path

def load_bank_npz(part: str, train_tag: str, method: str) -> torch.Tensor:
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_bank_train.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    B = torch.from_numpy(data["bank"].astype(np.float32))
    print(f"[npz] loaded bank: {B.shape} from {path}")
    return B

# ================= SCORING =================
@torch.no_grad()
def score_image_nn_faiss(F_img: torch.Tensor, bank_cpu: torch.Tensor, tile_q: int = 65536) -> torch.Tensor:
    C, Hf, Wf = F_img.shape
    Q = F_img.permute(1,2,0).reshape(-1, C).contiguous().cpu().numpy().astype(np.float32)
    B = bank_cpu.contiguous().cpu().numpy().astype(np.float32)
    cpu_index = faiss.IndexFlatL2(C)
    use_gpu = _FAISS_OK and faiss.get_num_gpus() > 0
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index
    index.add(B)
    L = Q.shape[0]
    out = np.empty((L,), dtype=np.float32)
    for q0 in range(0, L, tile_q):
        q1 = min(q0 + tile_q, L)
        D, _ = index.search(Q[q0:q1], k=1)
        out[q0:q1] = D[:,0]
    if use_gpu:
        faiss.index_gpu_to_cpu(index)
    return torch.from_numpy(out.reshape(Hf, Wf))

@torch.no_grad()
def score_image_nn_torch(F_img: torch.Tensor, bank_cpu: torch.Tensor, tile_q: int = 4096, tile_b: int = 20000, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> torch.Tensor:
    C, Hf, Wf = F_img.shape
    Q = F_img.permute(1,2,0).reshape(-1, C).to(device)
    out = torch.full((Q.shape[0],), float("inf"), dtype=torch.float32, device=device)
    for q0 in range(0, Q.shape[0], tile_q):
        q1 = min(q0 + tile_q, Q.shape[0])
        q_blk = Q[q0:q1]
        best = torch.full((q1-q0,), float("inf"), dtype=torch.float32, device=device)
        for b0 in range(0, bank_cpu.shape[0], tile_b):
            b1 = min(b0 + tile_b, bank_cpu.shape[0])
            Bt = bank_cpu[b0:b1].to(device, non_blocking=True)
            d = torch.cdist(q_blk, Bt)
            m,_ = d.min(dim=1)
            best = torch.minimum(best, m)
            del Bt, d, m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        out[q0:q1] = best
    return out.view(Hf, Wf).cpu()

# ================= MAIN =================

def main():
    set_seeds(SEED)
    device = torch.device(DEVICE)

    # Data
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
    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=16, device=device)

    # Model
    model = ShallowWRN(FEATURE_LAYER, device)

    # === TRAIN features ===
    try:
        payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        F_tr = payload["F_tr"]   # (N,C,H,W)
        print("[cache] loaded train features.")
    except FileNotFoundError:
        Fs = []
        for x,_,_ in tqdm(train_loader, desc="| feature extraction | train |", leave=False):
            f,_ = model.forward_features(x.to(device, non_blocking=True))
            Fs.append(f.detach().cpu())
        F_tr = torch.cat(Fs, 0)
        save_split_pickle({"F_tr":F_tr}, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    N,C,H,W = F_tr.shape

    # L2 norm sui canali (niente positional embedding nella versione fedele)
    F_tr_dev = l2norm(F_tr.to(device), dim=1)

    # Build channels
    bank_loaded = False
    if LOAD_CHANNELS:
        try:
            _ = load_channels_npz(CODICE_PEZZO, TRAIN_TAG, METHOD)
            bank_cpu = load_bank_npz(CODICE_PEZZO, TRAIN_TAG, METHOD)
            bank_loaded = True
        except FileNotFoundError:
            bank_loaded = False

    if not bank_loaded:
        chans, bank_cpu = build_channels_official(
            F_train=F_tr_dev,
            assoc_depth=ASSOC_DEPTH_D,
            search_rad=SEARCH_RAD,
            stride_h=STRIDE_H, stride_w=STRIDE_W,
            sim_min=SIM_MIN,
            min_span=MIN_CHANNEL_SPAN,
            max_channel_std=MAX_CHANNEL_STD,
            shuffle_seeds=SHUFFLE_SEEDS,
        )
        if bank_cpu.shape[0] == 0:
            raise RuntimeError("Nominal model (bank) empty: relax MAX_CHANNEL_STD / MIN_CHANNEL_SPAN or check SEARCH_RAD.")
        if SAVE_CHANNELS:
            save_channels_npz(chans, CODICE_PEZZO, TRAIN_TAG, METHOD)
            save_bank_npz(bank_cpu, CODICE_PEZZO, TRAIN_TAG, METHOD)

    # === VAL features ===
    Fs_val, ys = [], []
    for x,y,_ in tqdm(val_loader, desc="| feature extraction | val |", leave=False):
        f,_ = model.forward_features(x.to(device, non_blocking=True))
        Fs_val.append(f.detach())
        ys.extend([int(t.item()) for t in y])
    F_val_raw = torch.cat(Fs_val, 0)

    F_val = l2norm(F_val_raw.to(device), dim=1)
    _, _, Hf, Wf = F_val.shape

    # Inference
    raw_maps, img_scores = [], []
    gt_np = np.asarray(ys, dtype=np.int32)
    with torch.inference_mode():
        total = F_val.shape[0]
        pbar = tqdm(total=total, desc="| InReaCh (official) inference |", leave=False)
        for i in range(total):
            f = F_val[i]
            if USE_FAISS and _FAISS_OK:
                dist_map = score_image_nn_faiss(f, bank_cpu, tile_q=TILE_Q)
            else:
                dist_map = score_image_nn_torch(f, bank_cpu, tile_q=4096, tile_b=TILE_B, device=device)
            M = dist_map.unsqueeze(0).unsqueeze(0)
            mup = F.interpolate(M, size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            mup = gaussian_filter(mup, sigma=GAUSS_SIGMA).astype(np.float32)
            raw_maps.append(mup)
            img_scores.append(float(mup.max()))
            pbar.update(1)
        pbar.close()

    img_scores = np.array(img_scores, dtype=np.float32)

    # Image-level metrics (skippa se single-class)
    classes = np.unique(gt_np)
    if classes.size < 2:
        print("[image-level] skipped ROC-AUC (single-class validation). "
              "Add some good images to val to enable image-level metrics.")
        tn = fp = fn = tp = 0
        print(f"[val split] images={len(gt_np)} | good={(gt_np==0).sum()} | fault={(gt_np==1).sum()}")
    else:
        fpr, tpr, thr = roc_curve(gt_np, img_scores)
        auc_img = roc_auc_score(gt_np, img_scores)
        J = tpr - fpr
        best_idx = int(np.argmax(J)); best_thr = float(thr[best_idx])
        preds = (img_scores >= best_thr).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0,1]).ravel()
        print(f"[image-level] AUC={auc_img:.3f}  thr(Youden)={best_thr:.6f}  TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    # Pixel-level evaluation
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
