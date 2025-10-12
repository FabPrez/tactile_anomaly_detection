import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from data_loader import build_ad_datasets, make_loaders, save_split_pickle, load_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report
from view_utils import show_validation_grid_from_loader
# from view_utils import show_dataset_images  # se VIS_VALID_DATASET=True

# ----------------- CONFIG (paper-like) -----------------
METHOD = "PNI_PAPER"       # nuovo tag per non riusare vecchi pickle
CODICE_PEZZO = "PZ1"
TRAIN_POSITIONS = ["pos1", "pos2"]
VAL_GOOD_PER_POS = 20
VAL_GOOD_SCOPE  = ["pos1", "pos2"]
VAL_FAULT_SCOPE = ["pos1", "pos2"]
GOOD_FRACTION = 0.2

IMG_SIZE = 224
SEED = 42
BATCH_SIZE = 32
GAUSSIAN_SIGMA = 2
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True

# ---- PNI (paper) scelte pratiche ----
FEATURE_LAYER_IDX = 1      # 0=layer1, 1=layer2, 2=layer3 (usiamo layer2)
PATCH_P = 9                # vicinato p×p (paper: 9)
CDIST_SIZE = 2048          # |Cdist|
CEMB_SUBSAMPLE_PIX = 200_000  # quanti pixel campionare per costruire Cemb (~1%)
TAU = None                 # soglia T_tau; se None -> 1/(2*|Cemb|)
TEMP_SCALE = 2.0           # temperature scaling per MLP
LAMBDA = 1.0               # in exp(-lambda * ||phi-c||^2)
KPOS_IMG_SCORE = 5         # Top-K per image-level score (facoltativo)

# ---- MLP (vicinato) ----
MLP_LAYERS = 10
MLP_WIDTH  = 2048
MLP_EPOCHS = 10
MLP_LR     = 1e-3
MLP_WD     = 1e-4
MLP_BATCH  = 2048          # batch in #patch (non in immagini)

# ---- Refine (opzionale) ----
USE_REFINE_HEAD = False
REFINE_WEIGHTS  = None

# ---- image-level pooling ----
IMG_SCORE_POOL = "mean"     # "mean" | "max" | "p99"
# -------------------------------------------------------


# ============== Utils di base ==============
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


# ============== Estrazione feature (come tuo) ==============
def extract_features(model, loader, device, layers):
    outputs = []
    def hook(_m,_i,o): outputs.append(o)
    handles = [layer.register_forward_hook(hook) for layer in layers]
    feats = [[] for _ in layers]
    for x, y, m in tqdm(loader, desc="| feature extraction |"):
        x = x.to(device, non_blocking=True)
        with torch.no_grad(): _ = model(x)
        for i in range(len(layers)):
            feats[i].append(outputs[i].detach().cpu())
        outputs.clear()
    for i in range(len(layers)):
        feats[i] = torch.cat(feats[i], dim=0)
    for h in handles: h.remove()
    return feats


# ============== CORESET (farthest-first, semplice) ==============
@torch.no_grad()
def coreset_farthest_first(X: torch.Tensor, m: int) -> torch.Tensor:
    """
    X: (N, D) on CPU/float32
    Ritorna indici (m,) dei centroidi scelti.
    """
    N = X.size(0)
    m = min(m, N)
    sel = [np.random.randint(0, N)]
    # distanza minima corrente da set selezionato
    dmin = torch.cdist(X[sel], X)[0]  # (N,)
    for _ in range(1, m):
        i = torch.argmax(dmin).item()
        sel.append(i)
        dmin = torch.minimum(dmin, torch.cdist(X[i:i+1], X)[0])
    return torch.tensor(sel, dtype=torch.long)


# ============== MLP vicinato ==============
class MLPNeighbor(nn.Module):
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
        if T != 1.0:
            logits = logits / T
        return F.softmax(logits, dim=1)


# ============== Refine head (stub, opzionale) ==============
class RefineHead(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size
        self.block1 = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(True),
            nn.Conv2d(16,16,3,padding=1), nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16,8,3,padding=1), nn.ReLU(True),
            nn.Conv2d(8,1,1)
        )
    def forward(self, s):
        x = F.interpolate(s, size=self.out_size, mode='bilinear', align_corners=False)
        x = self.block1(x); x = self.block2(x)
        return x


# ============== Helper: estrazione patch centro + vicinato ==============
def build_center_and_neighbor_vectors(feat: torch.Tensor, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    feat: (N, C, H, W)  (CPU float32)
    Ritorna:
      centers:  (N*H*W, C)
      neighvec: (N*H*W, (p*p - 1)*C)  # vicinato flatten, SENZA il centro
    """
    N, C, H, W = feat.shape
    pad = p // 2
    # unfold per patch p×p
    patches = F.unfold(feat, kernel_size=p, padding=pad, stride=1)  # (N, C*p*p, H*W)
    patches = patches.transpose(1,2).contiguous()  # (N, H*W, C*p*p)
    patches = patches.view(N*H*W, C*p*p)          # (N*H*W, C*p*p)
    # centro e vicinato
    mid = (p*p)//2
    patches_resh = patches.view(-1, p*p, C)       # (N*H*W, p*p, C)
    centers = patches_resh[:, mid, :]             # (N*H*W, C)
    neigh   = torch.cat([patches_resh[:, :mid, :], patches_resh[:, mid+1:, :]], dim=1)  # (N*H*W, p*p-1, C)
    neigh   = neigh.reshape(neigh.shape[0], -1)   # (N*H*W, (p*p-1)*C)
    return centers, neigh


# ============== Costruzione Cemb / Cdist + MLP + Prior ==============
def build_pni_components_from_train(train_feat: torch.Tensor, device: torch.device):
    """
    train_feat: (N, C, H, W)  (CPU)
    Costruisce:
      - Cemb (subset dai centri dei patch) [Ncemb, C]
      - Cdist (coreset da Cemb)            [CDIST_SIZE, C]
      - mlp (addestrato) che approssima p(cdist | Np(x))
      - hist prior per posizione p(cdist | x): (H, W, CDIST_SIZE)
      - tau
    Restituisce dict con tutto (CPU salvo mlp su device).
    """
    N, C, H, W = train_feat.shape
    print(f"[PNI] train_feat layer2: N={N} C={C} HxW={H}x{W}")

    # 1) Centri + vicinato
    centers, neigh = build_center_and_neighbor_vectors(train_feat, PATCH_P)  # (N*H*W, C), (N*H*W, (p*p-1)*C)

    # 2) Cemb via coreset su campione (per velocità)
    total_pix = centers.shape[0]
    samp = min(CEMB_SUBSAMPLE_PIX, total_pix)
    samp_idx = torch.randperm(total_pix)[:samp]
    centers_samp = centers[samp_idx].contiguous()  # (samp, C)
    print(f"[PNI] Cemb: sampling {samp}/{total_pix} pixels for coreset...")
    cemb_idx = coreset_farthest_first(centers_samp, m=min(samp, 4*CDIST_SIZE))  # un po' più grande di Cdist
    Cemb = centers_samp[cemb_idx].contiguous()                                   # (Ncemb, C)
    Ncemb = Cemb.shape[0]
    print(f"[PNI] |Cemb| = {Ncemb}")

    # 3) Cdist via coreset su Cemb (-> 2048)
    print(f"[PNI] building Cdist ({CDIST_SIZE}) from Cemb...")
    cdist_idx = coreset_farthest_first(Cemb, m=min(CDIST_SIZE, Ncemb))
    Cdist = Cemb[cdist_idx].contiguous()  # (CDIST_SIZE, C)
    K = Cdist.shape[0]
    print(f"[PNI] |Cdist| = {K}")

    # 4) Target: mappa ogni centro al cdist più vicino (indice)
    with torch.no_grad():
        targ = []
        BS = 20000
        Cdist_cuda = Cdist.to(device)
        for s in range(0, centers.shape[0], BS):
            c = centers[s:s+BS].to(device)
            d = torch.cdist(c, Cdist_cuda)  # (bs, K)
            targ.append(torch.argmin(d, dim=1).cpu())
        target_cdist_idx = torch.cat(targ, dim=0)  # (N*H*W,)

    # 5) MLP training per p(cdist | Np(x))
    in_dim  = neigh.shape[1]
    out_dim = K
    print(f"[PNI] training MLP: in={in_dim}, out={out_dim}, layers={MLP_LAYERS}, width={MLP_WIDTH}")
    mlp = MLPNeighbor(in_dim, out_dim, width=MLP_WIDTH, layers=MLP_LAYERS).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=MLP_LR, weight_decay=MLP_WD)

    # dataset di patch (sotto-campioniamo per velocità)
    NUM_PATCH_TRAIN = min(1_000_000, neigh.shape[0])  # fino a 1M patch
    patch_idx = torch.randperm(neigh.shape[0])[:NUM_PATCH_TRAIN]
    neigh_train = neigh[patch_idx]
    targ_train  = target_cdist_idx[patch_idx]

    mlp.train()
    for ep in range(MLP_EPOCHS):
        perm = torch.randperm(NUM_PATCH_TRAIN)
        losses = []
        for s in range(0, NUM_PATCH_TRAIN, MLP_BATCH):
            idx = perm[s:s+MLP_BATCH]
            xb = neigh_train[idx].to(device)
            yb = targ_train[idx].to(device)
            logits = mlp.net(xb) / TEMP_SCALE
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[MLP] epoch {ep+1}/{MLP_EPOCHS} loss={np.mean(losses):.4f}")
    mlp.eval()

    # 6) Prior posizionale p(cdist | x) come istogramma per (h,w)
    print("[PNI] building positional prior histogram...")
    HWIN = H * W
    hist = torch.zeros((HWIN, K), dtype=torch.float32)
    for i in tqdm(range(N), desc="| prior hist |"):
        block = target_cdist_idx[i*HWIN:(i+1)*HWIN]  # (HW,)
        hist[torch.arange(HWIN), block] += 1.0
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-12)  # (HW, K)
    hist = hist.view(H, W, K).contiguous()                 # (H,W,K)

    # 7) Tau
    tau = (1.0 / (2.0 * float(Ncemb))) if (TAU is None) else float(TAU)
    print(f"[PNI] tau = {tau:.6e} (|Cemb|={Ncemb})")

    payload = {
        "layer_idx": FEATURE_LAYER_IDX,
        "shape": (H, W),
        "Cemb": Cemb,              # (Ncemb, C) CPU
        "Cdist": Cdist,            # (K, C) CPU
        "hist": hist,              # (H, W, K) CPU
        "tau": float(tau),
        "mlp_state": mlp.state_dict(),  # su CPU-friendly
        "cfg": {
            "PATCH_P": PATCH_P,
            "TEMP_SCALE": TEMP_SCALE,
            "LAMBDA": LAMBDA,
        }
    }
    return payload


# ============== Inference PNI (paper) ==============
@torch.no_grad()
def pni_infer_score_map(test_feat: torch.Tensor, payload: dict, device: torch.device) -> np.ndarray:
    """
    test_feat: (1, C, H, W) CPU (layer scelto)
    payload: dict con Cdist/hist/mlp/tau...
    Ritorna mappa (IMG_SIZE, IMG_SIZE) numpy.
    """
    H, W = payload["shape"]
    Cdist = payload["Cdist"].to(device)            # (K, C)
    hist  = payload["hist"]                        # (H, W, K) CPU
    tau   = payload["tau"]
    P = PATCH_P
    K = Cdist.shape[0]
    C = test_feat.shape[1]

    # ricostruisci MLP
    mlp = MLPNeighbor(in_dim=(P*P-1)*C, out_dim=K, width=MLP_WIDTH, layers=MLP_LAYERS).to(device).eval()
    mlp.load_state_dict(payload["mlp_state"], strict=True)

    # estrai centro + vicinato (1*H*W, C), (1*H*W, (P*P-1)*C)
    centers, neigh = build_center_and_neighbor_vectors(test_feat, P)  # CPU
    # p(c | Np(x)) via MLP (temperature scaling)
    probs_neigh = mlp(neigh.to(device), T=TEMP_SCALE)      # (HW, K)

    # p(c | x) via hist
    probs_pos = hist.view(-1, K).to(device)                # (HW, K)

    # combinazione (media) + sogliatura T_tau
    probs = 0.5 * (probs_neigh + probs_pos)               # (HW, K)
    keep = (probs > tau).float()
    probs = probs * keep                                   # filtra i c poco probabili

    # p(phi|c) ~ exp(-||phi-c||^2)
    centers = centers.to(device)                           # (HW, C)
    BS = 8192
    log_prob_phi_given_Omega = torch.empty((H*W,), dtype=torch.float32, device=device)
    for s in range(0, H*W, BS):
        ce = centers[s:s+BS]                               # (bs, C)
        pr = probs[s:s+BS]                                 # (bs, K)
        d = torch.cdist(ce, Cdist)                         # (bs, K)
        p_phi_c = torch.exp(-LAMBDA * (d ** 2))            # (bs, K)
        post = p_phi_c * pr
        p_phi_Omega, _ = post.max(dim=1)                   # (bs,)
        log_prob_phi_given_Omega[s:s+BS] = torch.clamp(-torch.log(p_phi_Omega + 1e-12), min=0).float()

    score = log_prob_phi_given_Omega.view(1, 1, H, W)      # (1,1,H,W)
    score = F.interpolate(score, size=IMG_SIZE, mode='bilinear', align_corners=False)
    score = score.squeeze().detach().cpu().numpy()         # (IMG, IMG)
    if GAUSSIAN_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        score = gaussian_filter(score, sigma=GAUSSIAN_SIGMA)
    return score


# ============================== MAIN ==============================
def main():
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== DATA ======
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

    # ====== MODEL ======
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model   = wide_resnet50_2(weights=weights).to(device).eval()
    layers  = [model.layer1[-1], model.layer2[-1], model.layer3[-1]]

    # ====== FEATURE CACHE ======
    try:
        feats_train_all = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[cache] Train features loaded.")
    except FileNotFoundError:
        print("[cache] Extracting train features...")
        feats_train_all = extract_features(model, train_loader, device, layers)
        save_split_pickle(feats_train_all, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    try:
        val_pack = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)
        feats_val_all, gt_list = val_pack["features"], val_pack["labels"]
        print("[cache] Validation features loaded.")
    except FileNotFoundError:
        print("[cache] Extracting validation features...")
        feats_val_all = extract_features(model, val_loader, device, layers)
        gt_list = []
        for _, y, _ in val_loader: gt_list.extend(y.cpu().numpy())
        save_split_pickle({"features": feats_val_all, "labels": np.array(gt_list, dtype=np.int64)},
                          CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)

    # ====== SELEZIONA LAYER PER PNI ======
    feat_train = feats_train_all[FEATURE_LAYER_IDX].contiguous()  # (N, C, H, W) CPU
    feat_val   = feats_val_all[FEATURE_LAYER_IDX].contiguous()    # (M, C, H, W) CPU

    # ====== COSTRUISCI o CARICA COMPONENTI PNI PAPER (FIX split) ======
    COMP_METHOD = f"{METHOD}_COMP"   # es: "PNI_PAPER_COMP"
    try:
        # usa split="train" per compatibilità con data_loader
        pni_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=COMP_METHOD)
        print("[cache] PNI paper components loaded.")
    except FileNotFoundError:
        print("[cache] Building PNI paper components...")
        pni_payload = build_pni_components_from_train(feat_train, device=device)
        save_split_pickle(pni_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=COMP_METHOD)

    # ====== Refine head opzionale ======
    refine_head = None
    if USE_REFINE_HEAD:
        refine_head = RefineHead(IMG_SIZE).to(device).eval()
        if REFINE_WEIGHTS:
            sd = torch.load(REFINE_WEIGHTS, map_location=device)
            refine_head.load_state_dict(sd, strict=False)

    # ====== INFERENZA MAPPE ======
    score_map_list = []
    for idx in tqdm(range(feat_val.shape[0]), desc="| PNI (paper) inference |"):
        s = pni_infer_score_map(feat_val[idx:idx+1], pni_payload, device=device)
        # (opzionale) refine fusion 10% come nel paper (se hai pesi)
        if refine_head is not None:
            with torch.no_grad():
                t = torch.from_numpy(s)[None,None].to(device)
                r = refine_head(t).squeeze().detach().cpu().numpy()
            s = 0.9*s + 0.1*r
        score_map_list.append(s)

    # ====== IMAGE LEVEL SCORE ======
    img_scores = np.array([pool_image_score(sm, mode=IMG_SCORE_POOL) for sm in score_map_list])
    gt_np = np.asarray(gt_list, dtype=np.int32)
    fpr, tpr, thresholds = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")

    J = tpr - fpr; best_idx = int(np.argmax(J)); best_thr = float(thresholds[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0, 1]).ravel()
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    # ====== VISUAL ======
    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True, overlay=True, overlay_alpha=0.45
        )

    # ====== PIXEL-LEVEL EVAL ======
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
