# inreach_full_ce.py
# InReaCh "faithful-structure":
# - feature intermedie
# - image coreset (ancore)
# - inter-realization channels tramite associazione locale (search window)
# - filtro span/spread
# - modello canale-wise (mu, cov, Cholesky)
# - scoring = min Mahalanobis per patch
# - tiling per memoria
#
# Compatibile con: build_ad_datasets, make_loaders, load/save_split_pickle, run_pixel_level_evaluation

import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter

from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ----------------- CONFIG -----------------
METHOD = "INREACH_FULL_CE"
CODICE_PEZZO = "PZ3"

TRAIN_POSITIONS   = ["pos1","pos2"]
VAL_GOOD_PER_POS  = 20
VAL_GOOD_SCOPE    = ["pos1","pos2"]
VAL_FAULT_SCOPE   = ["pos1","pos2"]
GOOD_FRACTION     = 1.0

IMG_SIZE    = 224
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Scelta layer (intermedio, descrittivo)
FEATURE_LAYER = "layer2"    # layer2[-1] hook (WRN50-2)

# Core idea InReaCh
CORESET_IMGS   = 32         # # immagini ancora (via k-center su globali)
SEARCH_RAD     = 1          # raggio finestra locale (±1 => 3x3)
SIM_MIN        = 0.00       # soglia similitudine coseno per accettare match
STRIDE_H       = 1          # stride spaziale per canali (1 => tutte le posizioni)
STRIDE_W       = 1

# filtro span/spread
SPAN_MIN       = 0.60       # frazione minima di immagini con match valido
SPREAD_MAX     = 1.25       # varianza media massima per canale
EPS_COV        = 1e-3       # ridge per stabilità

# Inference
GAUSS_SIGMA    = 2.0
TILE_Q         = 4096       # query (pixel) per tile nel calcolo Mahalanobis
TILE_K         = 64         # canali per tile (Cholesky solve batched)

# Valutazione
USE_THRESHOLD  = "pro"
FPR_LIMIT      = 0.01
# ------------------------------------------


def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def get_backbone(device):
    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
    m.eval()
    outs = []
    def hook(_m,_i,o): outs.append(o)
    # useremo layer2 come in molte AD per patch descrittive
    m.layer2[-1].register_forward_hook(hook)
    m.avgpool.register_forward_hook(hook)  # per coreset immagini
    return m, outs


@torch.no_grad()
def extract_features(model, outs, loader, device):
    Fs, Gs = [], []
    for x,_,_ in tqdm(loader, desc="| feature extraction | InReaCh |"):
        _ = model(x.to(device, non_blocking=True))
        layer2, avg = outs[0], outs[1]; outs.clear()
        Fs.append(layer2.detach().cpu())               # (B,C,H,W)
        Gs.append(avg.detach().cpu())                  # (B,2048,1,1)
    F = torch.cat(Fs, 0)                               # (N,C,H,W)
    G = torch.flatten(torch.cat(Gs, 0), 1)             # (N,2048)
    return F, G


@torch.no_grad()
def kcenter_coreset(X: np.ndarray, m: int, device: torch.device):
    """Greedy k-center su globali: X (N,D) -> indici (m,)"""
    N = X.shape[0]
    if m >= N: return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(1234)
    sel = [int(rng.integers(0,N))]
    Xt = torch.from_numpy(X).to(device)
    centers = Xt[sel[-1]:sel[-1]+1]
    dmin = torch.cdist(Xt, centers).squeeze(1)  # (N,)
    for _ in tqdm(range(1, m), desc="| coreset imgs |"):
        idx = int(torch.argmax(dmin).item())
        sel.append(idx)
        c = Xt[idx:idx+1]
        dmin = torch.minimum(dmin, torch.cdist(Xt, c).squeeze(1))
    return np.array(sel, dtype=np.int64)


def cosine_sim(a: torch.Tensor, b: torch.Tensor):
    # a: (...,C), b:(...,C) -> sim coseno
    return (a*b).sum(dim=-1)


def best_match_in_window(feat_img: torch.Tensor, ref_vec: torch.Tensor,
                         h:int, w:int, rad:int) -> Tuple[int,int,float]:
    """
    feat_img: (C,H,W) L2-normalized
    ref_vec:  (C,)
    cerca nella finestra [h-r:h+r, w-r:w+r] la posizione con max cosine
    """
    C,H,W = feat_img.shape
    h0, h1 = max(0,h-rad), min(H-1,h+rad)
    w0, w1 = max(0,w-rad), min(W-1,w+rad)
    window = feat_img[:, h0:h1+1, w0:w1+1].permute(1,2,0).reshape(-1, C)  # (M,C)
    sims = cosine_sim(window, ref_vec.unsqueeze(0))                        # (M,)
    idx = int(torch.argmax(sims).item())
    hh = h0 + idx // (w1-w0+1)
    ww = w0 + idx %  (w1-w0+1)
    return hh, ww, float(sims[idx].item())


@dataclass
class Channel:
    mu:  torch.Tensor   # (C,)
    cov: torch.Tensor   # (C,C)
    L:   torch.Tensor   # Cholesky factor (C,C) per inferenza rapida
    span: float         # copertura
    spread: float       # var media
    # opzionale: info posizionale (ancora) per pruning rapido
    h: int
    w: int


def compute_channel_stats(stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    stack: (M,C) vettori normalizzati (cosine space)
    ritorna: mu(C,), cov(C,C) con ridge, spread (trace(cov)/C)
    """
    mu = stack.mean(dim=0)
    dif = stack - mu
    cov = (dif.t() @ dif) / max(1, stack.shape[0]-1)
    spread = torch.trace(cov) / stack.shape[1]
    cov = cov + EPS_COV*torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return mu, cov, float(spread.item())


def build_channels(F_train: torch.Tensor,
                   img_anchors: np.ndarray,
                   search_rad:int=1,
                   stride_h:int=1, stride_w:int=1,
                   sim_min:float=0.0) -> List[Channel]:
    """
    F_train: (N,C,H,W) feature L2-normalized
    img_anchors: indici immagini ancore (per copertura)
    Per ogni (anchor_img, h, w) (con stride), crea un canale associando match
    nelle altre immagini dentro finestra di raggio 'search_rad'.
    """
    N,C,H,W = F_train.shape
    chans: List[Channel] = []
    device = F_train.device

    for ia in tqdm(img_anchors, desc="| build channels |"):
        anchor = F_train[ia]             # (C,H,W)
        for h in range(0, H, stride_h):
            for w in range(0, W, stride_w):
                ref = anchor[:,h,w]      # (C,)
                acc = []
                valid = 0
                for j in range(N):
                    feat_j = F_train[j]
                    hh, ww, sim = best_match_in_window(feat_j, ref, h, w, search_rad)
                    if sim >= sim_min:
                        acc.append(feat_j[:, hh, ww])
                        valid += 1
                span = valid / float(N)
                if span < SPAN_MIN:
                    continue
                stack = torch.stack(acc, dim=0)   # (valid, C)
                mu, cov, spread = compute_channel_stats(stack)
                if spread <= SPREAD_MAX:
                    L = torch.linalg.cholesky(cov)
                    chans.append(Channel(mu=mu, cov=cov, L=L, span=span, spread=spread, h=h, w=w))
    return chans


@torch.no_grad()
def mahalanobis_min_over_channels(Q: torch.Tensor,
                                  chans: List[Channel],
                                  tile_k:int=TILE_K) -> torch.Tensor:
    """
    Q: (L,C) query L2-normalized (una immagine flattenata)
    Ritorna: dist_min (L,) = min_j (q - mu_j)^T Σ_j^{-1} (q - mu_j)
    Calcolo a tile sui canali con Cholesky precomputato.
    """
    device = Q.device
    Lq, C = Q.shape
    dist_min = torch.full((Lq,), float("inf"), dtype=torch.float32, device=device)

    # pack canali in tensori (per batch solve)
    mus  = torch.stack([c.mu  for c in chans], dim=0).to(device) if len(chans)>0 else torch.empty((0,C), device=device)
    Ls   = torch.stack([c.L   for c in chans], dim=0).to(device) if len(chans)>0 else torch.empty((0,C,C), device=device)

    for k0 in range(0, len(chans), tile_k):
        k1 = min(k0+tile_k, len(chans))
        mu_k  = mus[k0:k1]         # (K,C)
        L_k   = Ls[k0:k1]          # (K,C,C)

        # diffs per canale: (K, Lq, C)
        diffs = (Q.unsqueeze(0) - mu_k.unsqueeze(1))   # broadcast
        # cholesky_solve: risolvi Σ x = (q - mu)^T --> shape handling:
        # vogliamo, per ogni canale k: dist2(q) = (q-mu)^T Σ^{-1} (q-mu)
        # soluzione: per k, metti (C, Lq), solve con L_k (K,C,C) batched.
        diffs_T = diffs.permute(0,2,1).contiguous()    # (K,C,Lq)
        sol = torch.cholesky_solve(diffs_T, L_k)       # (K,C,Lq)
        dist2 = (diffs_T * sol).sum(dim=1)             # (K,Lq)
        dmin_k, _ = torch.min(dist2, dim=0)            # (Lq,)
        dist_min = torch.minimum(dist_min, dmin_k.float())

    return dist_min  # (L,)


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
        F_tr_raw, G_tr = extract_features(model, outs, train_loader, device)  # (N,C,H,W), (N,2048)
        F_tr = l2norm(F_tr_raw, dim=1).cpu()                                   # normalize canali
        save_split_pickle({"F_tr":F_tr, "G_tr":G_tr}, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    N,C,H,W = F_tr.shape

    # ===== IMG CORESEt (ancore) =====
    anchors = kcenter_coreset(G_tr.numpy().astype(np.float32),
                              m=min(CORESET_IMGS, G_tr.shape[0]),
                              device=device)

    # ===== BUILD CHANNELS =====
    try:
        chp = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD+"_channels")
        chans_np = chp["channels"]
        chans = []
        for d in chans_np:
            mu  = torch.from_numpy(d["mu"]).float()
            cov = torch.from_numpy(d["cov"]).float()
            L   = torch.from_numpy(d["L"]).float()
            chans.append(Channel(mu=mu, cov=cov, L=L, span=float(d["span"]), spread=float(d["spread"]),
                                 h=int(d["h"]), w=int(d["w"])))
        print(f"[cache] canali caricati: {len(chans)}")
    except FileNotFoundError:
        F_tr_dev = F_tr.to(device)
        chans = build_channels(F_tr_dev, anchors,
                               search_rad=SEARCH_RAD,
                               stride_h=STRIDE_H, stride_w=STRIDE_W,
                               sim_min=SIM_MIN)
        chans_np = [{"mu": c.mu.cpu().numpy(), "cov": c.cov.cpu().numpy(), "L": c.L.cpu().numpy(),
                     "span": c.span, "spread": c.spread, "h": c.h, "w": c.w} for c in chans]
        save_split_pickle({"channels": chans_np}, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD+"_channels")
        print(f"[build] canali creati: {len(chans)}")

    if len(chans) == 0:
        raise RuntimeError("Nessun canale valido dopo filtro span/spread. Allenta SPAN_MIN/SPREAD_MAX o aumenta SEARCH_RAD/CORESET_IMGS.")

    # ===== FEATURE (VAL) =====
    F_val_raw, _ = extract_features(model, outs, val_loader, device)  # (Nv,C,H,W)
    F_val = l2norm(F_val_raw, dim=1).to(device)

    # ===== INFERENCE =====
    raw_maps = []
    img_scores = []
    gt_list = []

    with torch.inference_mode():
        for (x,y,_), f in tqdm(zip(val_loader, F_val), total=F_val.shape[0], desc="| InReaCh inference |"):
            gt_list.extend(y.cpu().numpy())
            # f: (C,H,W)
            C,H,W = f.shape
            q = f.permute(1,2,0).reshape(-1, C)  # (L,C)

            # min Mahalanobis su canali (tile K)
            dist_min = mahalanobis_min_over_channels(q, chans, tile_k=TILE_K)  # (L,)
            M = dist_min.view(H,W).unsqueeze(0).unsqueeze(0)                   # (1,1,H,W)
            mup = F.interpolate(M, size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            mup = gaussian_filter(mup, sigma=GAUSS_SIGMA).astype(np.float32)
            raw_maps.append(mup)
            img_scores.append(float(mup.max()))

    img_scores = np.array(img_scores, dtype=np.float32)
    gt_np = np.asarray(gt_list, dtype=np.int32)

    # ----- image-level breve report -----
    fpr, tpr, thr = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    J = tpr - fpr
    best_idx = int(np.argmax(J)); best_thr = float(thr[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0,1]).ravel()
    print(f"[image-level] AUC={auc_img:.3f}  thr(Youden)={best_thr:.6f}  TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    # ----- pixel-level con le tue utility -----
    results = run_pixel_level_evaluation(
        score_map_list=raw_maps,
        val_set=val_set,
        img_scores=img_scores,
        use_threshold=USE_THRESHOLD,
        fpr_limit=FPR_LIMIT,
        vis=True,
        vis_ds_or_loader=val_loader.dataset
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")


if __name__ == "__main__":
    main()
