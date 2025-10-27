# inreach_official_like_faiss_tiledready.py
# Pipeline fedele al paper/repo, con ottimizzazioni opzionali:
# - limitazione ancore (CORESET_IMGS) oppure ALL
# - stride sulle posizioni (STRIDE_H/STRIDE_W)
# - FAISS GPU se disponibile, altrimenti CPU o fallback torch.cdist
# - salvataggi canali/bank disattivabili per evitare I/O costosi

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

# accelera conv/unfold
torch.backends.cudnn.benchmark = True

# ---- prova ad importare FAISS ----
USE_FAISS = True
_FAISS_OK = False
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

# --- i tuoi pacchetti ---
from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# ----------------- CONFIG -----------------
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

# Backbone/layer
FEATURE_LAYER = "layer2"    # hook a layer2[-1]

# Coreset immagini (ancore)
# - "ALL" usa tutte le immagini di train
# - un intero m seleziona m ancore via k-center (greedy)
CORESET_IMGS   = 64     # es.: 64 o 128 per accelerare

# Matching locale per canali
SEARCH_RAD     = 1          # 3x3
SIM_MIN        = 0.0
STRIDE_H       = 1          # ↑ (es. 2) = più veloce, ↓ accuratezza
STRIDE_W       = 1

# Filtri canale (purista: disattivati)
SPAN_MIN       = 0.0
SPREAD_MAX     = float("inf")

# Limite opzionale patch per canale (RAM). Purista: None
BANK_PER_CHANNEL_LIMIT = None

# Inference NN L2
TILE_Q         = 65536      # query per batch (per FAISS si può salire parecchio)
TILE_B         = 20000      # usato solo nel fallback torch.cdist
GAUSS_SIGMA    = 2.0

# I/O feature bank/channels
SAVE_CHANNELS  = False      # True = salva npz canali/bank (più lento ma riusabile)
LOAD_CHANNELS  = False      # True = prova a caricare se esistono
# ----------------------------------------------------


# ================== UTIL BASE ==================
def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def get_backbone(device):
    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
    m.eval()
    outs = []
    def hook(_m,_i,o): outs.append(o)
    m.layer2[-1].register_forward_hook(hook)  # feature intermedie
    m.avgpool.register_forward_hook(hook)     # globali per coreset (qui "ALL")
    return m, outs


@torch.no_grad()
def extract_features(model, outs, loader, device):
    Fs, Gs = [], []
    for x,_,_ in tqdm(loader, desc="| feature extraction | InReaCh |", leave=False):
        _ = model(x.to(device, non_blocking=True))
        layer2, avg = outs[0], outs[1]; outs.clear()
        Fs.append(layer2.detach().cpu())               # (B,C,H,W)
        Gs.append(avg.detach().cpu())                  # (B,2048,1,1)
    F = torch.cat(Fs, 0)                               # (N,C,H,W)
    G = torch.flatten(torch.cat(Gs, 0), 1)             # (N,2048)
    return F, G


@torch.no_grad()
def kcenter_coreset(X: np.ndarray, m, device: torch.device):
    # con m == "ALL" restituiamo tutti gli indici (purista)
    N = X.shape[0]
    if (m == "ALL") or (m is None) or (isinstance(m, str) and m.upper()=="ALL") or (isinstance(m,int) and m>=N):
        return np.arange(N, dtype=np.int64)
    # greedy k-center
    rng = np.random.default_rng(1234)
    sel = [int(rng.integers(0,N))]
    Xt = torch.from_numpy(X).to(device)
    centers = Xt[sel[-1]:sel[-1]+1]
    dmin = torch.cdist(Xt, centers).squeeze(1)
    for _ in tqdm(range(1, m), desc="| coreset imgs |", leave=False):
        idx = int(torch.argmax(dmin).item())
        sel.append(idx)
        c = Xt[idx:idx+1]
        dmin = torch.minimum(dmin, torch.cdist(Xt, c).squeeze(1))
    return np.array(sel, dtype=np.int64)


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
) -> Tuple[List[ChannelMeta], torch.Tensor]:

    if isinstance(img_anchors, np.ndarray):
        img_anchors = torch.from_numpy(img_anchors).long()
    device = F_train.device
    N, C, H, W = F_train.shape
    ksize = 2*search_rad + 1
    pad = search_rad

    # grid con lo stride
    hs = torch.arange(0, H, device=device, dtype=torch.long)[::stride_h]
    ws = torch.arange(0, W, device=device, dtype=torch.long)[::stride_w]
    Hs, Ws = torch.meshgrid(hs, ws, indexing="ij")                # (H', W')
    coords = torch.stack([Hs.reshape(-1), Ws.reshape(-1)], 1)     # (L, 2)
    L = coords.shape[0]

    # indici lineari per selezionare posizioni dall'unfold (stride=1)
    lin_ids_full = (torch.arange(H, device=device).unsqueeze(1) * W +
                    torch.arange(W, device=device))               # (H, W)
    l_ids = lin_ids_full[hs][:, ws].reshape(-1)                   # (L,)

    chans: List[ChannelMeta] = []
    bank_chunks: List[torch.Tensor] = []

    for ia in tqdm(img_anchors.tolist(), desc="| build channels |", leave=False):
        A = F_train[ia]                                           # (C,H,W)
        # vettori di riferimento dell'ancora alle sole posizioni (stride)
        A_vecs = A.permute(1,2,0)[hs][:, ws, :].reshape(-1, C).contiguous()   # (L, C)

        # accumulatore per posizione (liste di vettori C)
        per_pos_acc = [[] for _ in range(L)]

        # prealloc per ridurre malloc nel loop
        for j in range(N):
            Y = F_train[j:j+1]                                    # (1,C,H,W)
            patches = F.unfold(Y, kernel_size=ksize, padding=pad, stride=1)   # (1, C*K, H*W)
            patches_sel = patches[:, :, l_ids]                                   # (1, C*K, L)
            patches_sel = patches_sel.view(1, C, ksize*ksize, L).squeeze(0)      # (C, K, L)

            # sim coseno = dot product (già L2 sui canali)
            # A_vecs: (L,C), patches_sel: (C,K,L) -> sims: (K,L)
            sims = torch.einsum('lc,ckl->kl', A_vecs, patches_sel)               # (K, L)
            vals, argk = torch.max(sims, dim=0)                                  # (L,)
            valid_mask = (vals >= sim_min)
            if valid_mask.any():
                # gather corretto: index ha le stesse dims dell'input lungo dim=1
                idx = argk.view(1, 1, -1).expand(C, 1, L)                        # (C,1,L)
                best_vecs = torch.gather(patches_sel, dim=1, index=idx)          # (C,1,L)
                best_vecs = best_vecs.squeeze(1).t().contiguous()                # (L,C)

                l_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                for l_id in l_idx.tolist():
                    per_pos_acc[l_id].append(best_vecs[l_id:l_id+1].detach())    # (1,C)

            del Y, patches, patches_sel, sims, vals, argk, valid_mask, idx, best_vecs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # valutazione del canale (per posizione)
        for l_id, acc_list in enumerate(per_pos_acc):
            valid = len(acc_list)
            if valid == 0:
                continue
            span = valid / float(N)
            if span < span_min:
                continue

            stack = torch.cat(acc_list, dim=0)                                   # (valid, C) device
            spr = compute_spread(stack)
            if spr > spread_max:
                continue

            if per_channel_limit is not None and stack.shape[0] > per_channel_limit:
                sel = torch.randperm(stack.shape[0], device=stack.device)[:per_channel_limit]
                stack = stack[sel]

            bank_chunks.append(stack.cpu().float())
            h, w = int(coords[l_id, 0].item()), int(coords[l_id, 1].item())
            chans.append(ChannelMeta(h=h, w=w, span=float(span), spread=float(spr)))

        del A, A_vecs, per_pos_acc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # bank finale = unione patch nominali dei canali accettati (CPU float32)
    if len(bank_chunks) == 0:
        bank = torch.empty((0, C), dtype=torch.float32)
    else:
        bank = torch.cat(bank_chunks, dim=0).contiguous()

    return chans, bank


# ================== NOMINAL MODEL I/O ==================
def _features_dir_for(part: str, train_tag: str) -> str:
    base = os.path.join(os.path.dirname(__file__), "Dataset", part, "features", train_tag)
    os.makedirs(base, exist_ok=True)
    return base

def save_channels_npz(chans: List[ChannelMeta], part: str, train_tag: str, method: str):
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
    print(f"[npz] salvati canali: {path}")
    return path

def load_channels_npz(part: str, train_tag: str, method: str) -> List[ChannelMeta]:
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_channels_train.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    h = data["h"].astype(np.int32); w = data["w"].astype(np.int32)
    span = data["span"].astype(np.float32); spread = data["spread"].astype(np.float32)
    chans = [ChannelMeta(h=int(h[i]), w=int(w[i]), span=float(span[i]), spread=float(spread[i])) for i in range(h.shape[0])]
    print(f"[npz] caricati {len(chans)} canali da {path}")
    return chans

def save_bank_npz(bank: torch.Tensor, part: str, train_tag: str, method: str):
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_bank_train.npz")
    np.savez_compressed(path, bank=bank.numpy().astype(np.float16))
    print(f"[npz] salvato bank: {path}  (vecs={bank.shape[0]}, dim={bank.shape[1]})")
    return path

def load_bank_npz(part: str, train_tag: str, method: str) -> torch.Tensor:
    d = _features_dir_for(part, train_tag)
    path = os.path.join(d, f"{method.lower()}_bank_train.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    B = torch.from_numpy(data["bank"].astype(np.float32))   # (M,C) CPU
    print(f"[npz] caricato bank: {B.shape} da {path}")
    return B


# ================== SCORING (NN L2) ==================
@torch.no_grad()
def score_image_nn_faiss(
    F_img: torch.Tensor,      # (C,Hf,Wf) L2-normalized on channels (device-agnostic)
    bank_cpu: torch.Tensor,   # (M,C) CPU float32
    tile_q: int = 65536
) -> torch.Tensor:
    """
    1-NN L2 esatto con FAISS. Usa GPU se disponibile, altrimenti CPU.
    Output: dist_min (Hf, Wf) (torch.float32, CPU).
    """
    C, Hf, Wf = F_img.shape
    Q = F_img.permute(1,2,0).reshape(-1, C).contiguous().cpu().numpy().astype(np.float32)  # (L,C) np
    B = bank_cpu.contiguous().cpu().numpy().astype(np.float32)                              # (M,C) np

    # Indice L2 esatto
    cpu_index = faiss.IndexFlatL2(C)
    use_gpu = faiss.get_num_gpus() > 0
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        index = cpu_index
    index.add(B)  # build

    # query a blocchi
    L = Q.shape[0]
    out = np.empty((L,), dtype=np.float32)
    for q0 in range(0, L, tile_q):
        q1 = min(q0 + tile_q, L)
        D, _ = index.search(Q[q0:q1], k=1)  # D: (lq, 1) distanze L2
        out[q0:q1] = D[:, 0]

    if use_gpu:
        faiss.index_gpu_to_cpu(index)  # rilascia risorse GPU

    return torch.from_numpy(out.reshape(Hf, Wf))  # CPU


@torch.no_grad()
def score_image_nn_torch(
    F_img: torch.Tensor,      # (C,Hf,Wf) L2-normalized on channels
    bank_cpu: torch.Tensor,   # (M,C) CPU float32
    tile_q: int = 4096,
    tile_b: int = 20000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:
    """Fallback esatto con torch.cdist (più lento)."""
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
    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=16, device=device)

    # ===== MODEL =====
    model, outs = get_backbone(device)

    # ===== FEATURE (TRAIN) =====
    try:
        payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        F_tr = payload["F_tr"]; G_tr = payload["G_tr"]
        print("[cache] feature train caricate.")
    except FileNotFoundError:
        F_tr_raw, G_tr = extract_features(model, outs, train_loader, device)
        F_tr = l2norm(F_tr_raw, dim=1).cpu()
        save_split_pickle({"F_tr":F_tr, "G_tr":G_tr}, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    N,C,H,W = F_tr.shape

    # ===== IMAGE CORESEt (ancore) =====
    anchors = kcenter_coreset(
        G_tr.numpy().astype(np.float32),
        m=CORESET_IMGS,
        device=device
    )

    # ===== BUILD CHANNELS + NOMINAL MODEL =====
    bank_loaded = False
    chans = []
    if LOAD_CHANNELS:
        try:
            chans = load_channels_npz(CODICE_PEZZO, TRAIN_TAG, METHOD)
            bank_cpu = load_bank_npz(CODICE_PEZZO, TRAIN_TAG, METHOD)
            bank_loaded = True
        except FileNotFoundError:
            bank_loaded = False

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
            per_channel_limit=BANK_PER_CHANNEL_LIMIT
        )
        if bank_cpu.shape[0] == 0:
            raise RuntimeError("Nominal model (bank) vuoto: allenta filtri o controlla SEARCH_RAD/STRIDE.")
        if SAVE_CHANNELS:
            save_channels_npz(chans, CODICE_PEZZO, TRAIN_TAG, METHOD)
            save_bank_npz(bank_cpu, CODICE_PEZZO, TRAIN_TAG, METHOD)

    # ===== FEATURE (VAL) =====
    F_val_raw, _ = extract_features(model, outs, val_loader, device)  # (Nv,C,Hf,Wf)
    F_val = l2norm(F_val_raw, dim=1).to(device)
    _, C, Hf, Wf = F_val.shape

    # ===== INFERENCE =====
    raw_maps, img_scores, gt_list = [], [], []
    with torch.inference_mode():
        total_imgs, idx_feat = F_val.shape[0], 0
        pbar = tqdm(total=total_imgs, desc="| InReaCh (official+faiss) inference |", leave=False)

        for (x, y, _) in val_loader:
            Bsz = y.shape[0]
            f_batch = F_val[idx_feat:idx_feat+Bsz]; idx_feat += Bsz
            for b in range(Bsz):
                gt_list.append(int(y[b].item()))
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
    gt_np = np.asarray(gt_list, dtype=np.int32)

    # ----- image-level AUROC -----
    fpr, tpr, thr = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    J = tpr - fpr
    best_idx = int(np.argmax(J)); best_thr = float(thr[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0,1]).ravel()
    print(f"[image-level] AUC={auc_img:.3f}  thr(Youden)={best_thr:.6f}  TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

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
