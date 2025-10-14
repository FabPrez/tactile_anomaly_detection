# PNI_official_adapt.py — Adattamento fedele alla repo PNI (solo I/O), per la tua pipeline
import math
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from data_loader import build_ad_datasets, make_loaders, save_split_pickle, load_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report
from view_utils import show_validation_grid_from_loader

# ----------------- CONFIG -----------------
METHOD = "PNI_OFFICIAL"
CODICE_PEZZO = "PZ1"

# split come negli altri metodi
TRAIN_POSITIONS   = ["pos1","pos2"]
VAL_GOOD_PER_POS  = 0
VAL_GOOD_SCOPE    = ["pos1","pos2"]
VAL_FAULT_SCOPE   = ["pos1","pos2"]
GOOD_FRACTION     = 1.0

# backbone & feature extraction
IMG_SIZE  = 224
SEED      = 42
BATCH_SIZE = 32

# PNI core (fedeli alla repo/paper)
FEATURE_LAYER_IDX = 1          # 0=layer1, 1=layer2, 2=layer3 (di solito layer2)
PATCH_P      = 9               # vicinato p×p
CEMB_SUBSAMPLE_PIX = 200_000   # campione di pixel per coreset Cemb
CDIST_SIZE   = 2048            # |Cdist| (coreset da Cemb)
TAU          = None            # se None -> 1/(2*|Cemb|)
TEMP_SCALE   = 2.0             # temperature scaling per MLP
LAMBDA       = 1.0             # in exp(-lambda * ||phi - c||^2)

# MLP (come repo: classificatore del vicino cdist dato il vicinato)
MLP_LAYERS   = 10
MLP_WIDTH    = 2048
MLP_EPOCHS   = 10
MLP_LR       = 1e-3
MLP_WD       = 1e-4
MLP_BATCH    = 2048
MAX_PATCH_TRAIN = 1_000_000    # limite patch per il training MLP (per tempo/memoria)

# inference & visual
GAUSSIAN_SIGMA = 2
IMG_SCORE_POOL = "mean"        # "mean"|"max"|"p99" (pooling della heatmap per img-score)
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True
# ------------------------------------------


# ----------------- utils -----------------
def set_all_seeds(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def pool_image_score(arr, mode="mean"):
    if mode == "max": return float(np.max(arr))
    if mode == "p99": return float(np.percentile(arr, 99))
    return float(np.mean(arr))

@torch.no_grad()
def extract_layer_feats(model, loader, device, layers):
    """Estrae feature per i layer richiesti, cache-compatibile con la tua pipeline."""
    outputs = []
    def hook(_m,_i,o): outputs.append(o)
    hs = [layer.register_forward_hook(hook) for layer in layers]
    feats = [[] for _ in layers]
    for x, y, m in tqdm(loader, desc="| feature extraction |"):
        x = x.to(device, non_blocking=True)
        _ = model(x)
        for i in range(len(layers)):
            feats[i].append(outputs[i].detach().cpu())
        outputs.clear()
    for i in range(len(layers)): feats[i] = torch.cat(feats[i], dim=0)
    for h in hs: h.remove()
    return feats  # lista: [layer1, layer2, layer3]

def build_center_and_neighbor_vectors(feat: torch.Tensor, p: int):
    """
    feat: (N, C, H, W) CPU
    ritorna:
      centers: (N*H*W, C)
      neigh  : (N*H*W, (p*p-1)*C) senza il centro
    """
    N, C, H, W = feat.shape
    pad = p // 2
    patches = F.unfold(feat, kernel_size=p, padding=pad, stride=1)  # (N, C*p*p, H*W)
    patches = patches.transpose(1,2).contiguous().view(N*H*W, C*p*p) # (N*H*W, C*p*p)
    mid = (p*p)//2
    patches_resh = patches.view(-1, p*p, C)
    centers = patches_resh[:, mid, :]  # (N*H*W, C)
    neigh = torch.cat([patches_resh[:, :mid, :], patches_resh[:, mid+1:, :]], dim=1)  # (N*H*W, p*p-1, C)
    neigh = neigh.reshape(neigh.shape[0], -1)  # (N*H*W, (p*p-1)*C)
    return centers, neigh

@torch.no_grad()
def coreset_farthest_first(X: torch.Tensor, m: int) -> torch.Tensor:
    """Farthest-first su X (N,D) CPU float32. Ritorna indici (m,)."""
    N = X.size(0)
    m = min(m, N)
    if m <= 0: return torch.empty((0,), dtype=torch.long)
    sel = [np.random.randint(0, N)]
    dmin = torch.cdist(X[sel], X)[0]
    for _ in range(1, m):
        i = torch.argmax(dmin).item()
        sel.append(i)
        dmin = torch.minimum(dmin, torch.cdist(X[i:i+1], X)[0])
    return torch.tensor(sel, dtype=torch.long)

class MLPNeighbor(nn.Module):
    """MLP che approssima p(cdist | Np(x)); output softmax su |Cdist| classi."""
    def __init__(self, in_dim, out_dim, width=2048, layers=10):
        super().__init__()
        blocks = []
        d = in_dim
        for _ in range(layers-1):
            blocks += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.ReLU(True)]
            d = width
        blocks += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x, T=1.0):
        logits = self.net(x)
        if T != 1.0: logits = logits / T
        return F.softmax(logits, dim=1)


# ----------------- PNI: build components (repo-like) -----------------
def build_pni_components(train_feat: torch.Tensor, device: torch.device):
    """
    train_feat: (N, C, H, W) CPU (layer scelto)
    Ritorna un dict con:
      - Cemb (Ncemb, C)
      - Cdist (K, C)
      - hist  (H, W, K) prior posizionale p(c|x)
      - tau   (float)
      - mlp_state (state_dict)
      - cfg   (p, temp, lambda)
    """
    N, C, H, W = train_feat.shape
    print(f"[PNI] train layer: N={N} C={C} HxW={H}x{W}")

    # 1) centro + vicinato
    centers, neigh = build_center_and_neighbor_vectors(train_feat, PATCH_P)  # (N*HW,C), (N*HW,(p*p-1)C)
    total_pix = centers.shape[0]

    # 2) Cemb (coreset su campione di pixel   — fedele: coreset per ridurre ridondanza)
    samp = min(CEMB_SUBSAMPLE_PIX, total_pix)
    idx_samp = torch.randperm(total_pix)[:samp]
    centers_samp = centers[idx_samp].contiguous()
    print(f"[PNI] sampling for Cemb: {samp}/{total_pix}")
    cemb_idx = coreset_farthest_first(centers_samp.float(), m=min(samp, 4*CDIST_SIZE))
    Cemb = centers_samp[cemb_idx].contiguous()
    Ncemb = Cemb.shape[0]
    print(f"[PNI] |Cemb|={Ncemb}")

    # 3) Cdist (coreset da Cemb — fedele)
    cdist_idx = coreset_farthest_first(Cemb.float(), m=min(CDIST_SIZE, Ncemb))
    Cdist = Cemb[cdist_idx].contiguous()
    K = Cdist.shape[0]
    print(f"[PNI] |Cdist|={K}")

    # 4) Target per MLP: nearest Cdist index del centro (repo/paper)
    targ_chunks = []
    BS = 20000
    Cdist_dev = Cdist.to(device)
    with torch.no_grad():
        for s in range(0, centers.shape[0], BS):
            c = centers[s:s+BS].to(device)
            d = torch.cdist(c, Cdist_dev)        # (bs,K)
            targ_chunks.append(torch.argmin(d, dim=1).cpu())
    target_idx = torch.cat(targ_chunks, dim=0)   # (N*HW,)

    # 5) MLP su vicinato -> p(c | Np(x))  (cross-entropy sui target_idx)
    in_dim  = neigh.shape[1]
    out_dim = K
    mlp = MLPNeighbor(in_dim, out_dim, width=MLP_WIDTH, layers=MLP_LAYERS).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=MLP_LR, weight_decay=MLP_WD)

    NUM_PATCH = min(MAX_PATCH_TRAIN, neigh.shape[0])
    patch_idx = torch.randperm(neigh.shape[0])[:NUM_PATCH]
    neigh_train = neigh[patch_idx]
    targ_train  = target_idx[patch_idx]

    mlp.train()
    for ep in range(MLP_EPOCHS):
        perm = torch.randperm(NUM_PATCH)
        losses = []
        for s in range(0, NUM_PATCH, MLP_BATCH):
            idb = perm[s:s+MLP_BATCH]
            xb = neigh_train[idb].to(device)
            yb = targ_train[idb].to(device)
            logits = mlp.net(xb) / TEMP_SCALE
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[PNI][MLP] epoch {ep+1}/{MLP_EPOCHS} loss={np.mean(losses):.4f}")
    mlp.eval()

    # 6) Prior posizionale p(c|x): istogramma dei target_idx per (h,w)
    HWIN = H * W
    hist = torch.zeros((HWIN, K), dtype=torch.float32)
    for i in tqdm(range(N), desc="| prior hist |"):
        block = target_idx[i*HWIN:(i+1)*HWIN]   # (HW,)
        hist[torch.arange(HWIN), block] += 1.0
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-12)
    hist = hist.view(H, W, K).contiguous()

    # 7) Tau
    tau = (1.0 / (2.0 * float(Ncemb))) if TAU is None else float(TAU)
    print(f"[PNI] tau={tau:.6e}")

    payload = {
        "layer_idx": FEATURE_LAYER_IDX,
        "shape": (H, W),
        "Cemb": Cemb,                 # CPU
        "Cdist": Cdist,               # CPU
        "hist": hist,                 # CPU
        "tau": float(tau),
        "mlp_state": mlp.state_dict(),  # per ricostruire su device
        "cfg": {"PATCH_P": PATCH_P, "TEMP_SCALE": TEMP_SCALE, "LAMBDA": LAMBDA}
    }
    return payload


# ----------------- PNI: inference (repo-like) -----------------
@torch.no_grad()
def pni_infer_score_map(test_feat: torch.Tensor, payload: dict, device: torch.device) -> np.ndarray:
    """
    test_feat: (1, C, H, W) CPU (layer scelto)
    ritorna: heatmap (IMG_SIZE, IMG_SIZE) numpy, score = -log p(phi | Omega)
    """
    H, W = payload["shape"]
    Cdist = payload["Cdist"].to(device)          # (K,C)
    hist  = payload["hist"].view(-1, Cdist.shape[0]).to(device)  # (HW,K)
    tau   = payload["tau"]
    P     = payload["cfg"]["PATCH_P"]
    K     = Cdist.shape[0]
    C     = test_feat.shape[1]

    # ricostruisci MLP
    mlp = MLPNeighbor(in_dim=(P*P-1)*C, out_dim=K, width=MLP_WIDTH, layers=MLP_LAYERS).to(device).eval()
    mlp.load_state_dict(payload["mlp_state"], strict=True)

    # centro+vicinato
    centers, neigh = build_center_and_neighbor_vectors(test_feat, P)   # CPU
    centers = centers.to(device)                                       # (HW,C)
    probs_neigh = mlp(neigh.to(device), T=TEMP_SCALE)                  # (HW,K)
    probs_pos   = hist                                                 # (HW,K)

    # combinazione (media) + sogliatura tau
    probs = 0.5 * (probs_neigh + probs_pos)
    probs = probs * (probs > tau).float()

    # p(phi|c) ~ exp(-lambda * ||phi - c||^2) ; posterior = p(phi|c) * p(c | ...)
    BS = 8192
    neglog = torch.empty((H*W,), dtype=torch.float32, device=device)
    for s in range(0, H*W, BS):
        ce = centers[s:s+BS]                 # (bs,C)
        pr = probs[s:s+BS]                   # (bs,K)
        d  = torch.cdist(ce, Cdist)          # (bs,K)
        p_phi_c = torch.exp(-(LAMBDA * (d**2)))
        post = p_phi_c * pr
        p_phi_Omega, _ = post.max(dim=1)     # MAP su c
        neglog[s:s+BS] = torch.clamp(-torch.log(p_phi_Omega + 1e-12), min=0.0)

    score = neglog.view(1,1,H,W)
    score = F.interpolate(score, size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze().detach().cpu().numpy()
    if GAUSSIAN_SIGMA > 0:
        score = gaussian_filter(score, sigma=GAUSSIAN_SIGMA)
    return score


# ============================== MAIN ==============================
def main():
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

    # ======== DATA ========
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO, img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=SEED, transform=None,
    )
    TRAIN_TAG = meta["train_tag"]
    print(f"Train GOOD (pos {meta['train_positions']}): {meta['counts']['train_good']}")
    print(f"Val   GOOD: {meta['counts']['val_good']}")
    print(f"Val  FAULT (pos {meta['val_fault_positions']}): {meta['counts']['val_fault']}")
    print(f"Val  TOT: {meta['counts']['val_total']}")

    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=BATCH_SIZE, device=device)

    # ======== MODEL ========
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model   = wide_resnet50_2(weights=weights).to(device).eval()
    layers  = [model.layer1[-1], model.layer2[-1], model.layer3[-1]]

    # ======== FEATURE CACHE ========
    try:
        feats_train_all = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[cache] Train features loaded.")
    except FileNotFoundError:
        print("[cache] Extracting train features...")
        feats_train_all = extract_layer_feats(model, train_loader, device, layers)
        save_split_pickle(feats_train_all, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    try:
        val_pack = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)
        feats_val_all, gt_list = val_pack["features"], val_pack["labels"]
        print("[cache] Validation features loaded.")
    except FileNotFoundError:
        print("[cache] Extracting validation features...")
        feats_val_all = extract_layer_feats(model, val_loader, device, layers)
        gt_list = []
        for _, y, _ in val_loader: gt_list.extend(y.cpu().numpy())
        save_split_pickle({"features": feats_val_all, "labels": np.array(gt_list, dtype=np.int64)},
                          CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)

    # ======== SELEZIONA LAYER ========
    feat_train = feats_train_all[FEATURE_LAYER_IDX].contiguous()  # CPU
    feat_val   = feats_val_all[FEATURE_LAYER_IDX].contiguous()    # CPU

    # ======== COMPONENTI PNI (cache separata) ========
    COMP_METHOD = f"{METHOD}_COMP"
    try:
        pni_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=COMP_METHOD)
        print("[cache] PNI components loaded.")
    except FileNotFoundError:
        print("[cache] Building PNI components (official repo logic)...")
        pni_payload = build_pni_components(feat_train, device=device)
        save_split_pickle(pni_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=COMP_METHOD)

    # ======== INFERENZA MAPPE ========
    score_map_list = []
    for i in tqdm(range(feat_val.shape[0]), desc="| PNI (official) inference |"):
        s = pni_infer_score_map(feat_val[i:i+1], pni_payload, device=device)
        score_map_list.append(s)

    # ======== IMAGE-LEVEL SCORE ========
    img_scores = np.array([pool_image_score(sm, mode=IMG_SCORE_POOL) for sm in score_map_list], dtype=np.float32)
    gt_np = np.asarray(gt_list, dtype=np.int32)
    fpr, tpr, thresholds = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])
    preds_img = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds_img, labels=[0, 1]).ravel()
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    # Plot ROC
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={auc_img:.3f}")
    ax[0].plot([0,1],[0,1],'k--',linewidth=1)
    ax[0].set_title("Image-level ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
    plt.tight_layout(); plt.show()

    # Visual validation grid (opzionale)
    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds_img,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True, overlay=True, overlay_alpha=0.45
        )

    # ======== PIXEL-LEVEL EVAL ========
    results = run_pixel_level_evaluation(
        score_map_list=score_map_list,
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
