# fapm_official_adapted.py
# FAPM (Fast Adaptive Patch Memory) — adattato al dataset/utility di Valerio
# Riferimenti: FAPM_official (model.py/utils.py/test.py) + paper FAPM

import os, math
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# --- tue utility condivise (come in SPADE/PADIM) ---
from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from view_utils import show_dataset_images, show_validation_grid_from_loader
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.ndimage import gaussian_filter

# ======================================================
#                     CONFIG
# ======================================================
METHOD = "FAPM"
CODICE_PEZZO = "PZ3"

# Train/Val come nei tuoi script
TRAIN_POSITIONS = ["pos2"]
VAL_GOOD_PER_POS = 0
VAL_GOOD_SCOPE  = ["pos2"]   # "from_train" | "all_positions" | lista
VAL_FAULT_SCOPE = ["pos2"]   # "train_only" | "all" | lista
GOOD_FRACTION = 1.0

# FAPM (scelte aderenti alla repo/paper): 2 layer (l2, l3), banca patch per layer
IMG_SIZE = 224
SEED     = 42
GAUSSIAN_SIGMA = 4

# Coreset adattivo (patch-wise, greedy k-center). 1.0 = nessun coreset
CORESET_FRAC = 0.02          # es: 0.25 = tieni il 25% delle patch
CORESET_PROJ_D = None        # opzionale: None (niente RP), oppure int per random projection (es. 128)

# Visual
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True

# --- Batch unico per train e validation (richiesta tua) ---
BATCH_SIZE = 1

# ======================================================
#                 HELPER (stile minimo)
# ======================================================
def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def extract_backbone():
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model = wide_resnet50_2(weights=weights)
    model.eval()
    outputs = []
    def hook(_m, _in, out): outputs.append(out)
    # FAPM usa due scale (repo ufficiale/paper) → layer2 & layer3
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    return model, outputs

def feature_maps_to_patches(fm2: torch.Tensor, fm3: torch.Tensor):
    """
    Converte le due feature map (B,C,H,W) in liste di patch embeddings per layer:
      - upsample fm3 a fm2 (nearest, come matching patch-wise)
      - L2 norm canale
      - output: (B, H*W, C) per layer
    """
    # riallinea spatialmente l3 -> l2
    fm3u = F.interpolate(fm3, size=fm2.shape[-2:], mode='nearest')
    fm2 = l2norm(fm2, dim=1)
    fm3u = l2norm(fm3u, dim=1)

    B, C2, H, W = fm2.shape
    C3 = fm3u.shape[1]
    L = H * W
    p2 = fm2.permute(0, 2, 3, 1).reshape(B, L, C2).contiguous()
    p3 = fm3u.permute(0, 2, 3, 1).reshape(B, L, C3).contiguous()
    return p2, p3, (H, W)

def greedy_coreset_kcenter(X: torch.Tensor, m: int, seed=0):
    """
    Coreset adattivo patch-wise (greedy k-center) come in PatchCore/FAPM-style.
    X: (N, D) tensore su CPU/float32. Restituisce indici selezionati (m,).
    """
    N = X.shape[0]
    if m >= N or m <= 0:
        return np.arange(N, dtype=np.int64)

    rng = np.random.default_rng(seed=seed)
    first = int(rng.integers(low=0, high=N))
    selected = [first]
    dist_min = torch.cdist(X, X[first:first+1], p=2).squeeze(1)  # (N,)
    for _ in range(1, m):
        idx = int(torch.argmax(dist_min).item())
        selected.append(idx)
        d_new = torch.cdist(X, X[idx:idx+1], p=2).squeeze(1)
        dist_min = torch.minimum(dist_min, d_new)
    return np.array(selected, dtype=np.int64)

def maybe_random_project(X: np.ndarray, out_d: int, seed=123):
    """
    Proiezione casuale (Johnson–Lindenstrauss) solo per coreset speed/mem. Facoltativa.
    """
    if out_d is None or out_d <= 0 or out_d >= X.shape[1]:
        return X
    rng = np.random.default_rng(seed=seed)
    R = rng.standard_normal(size=(X.shape[1], out_d)).astype(np.float32) / math.sqrt(out_d)
    return (X @ R).astype(np.float32)

def top1_distance_per_patch(Q: torch.Tensor, Bank: torch.Tensor, block_b=4096):
    """
    Distanza L2 min per patch (NN) a blocchi.
      Q:   (Lq, C)  query patches (tensor, device=cpu/cuda)
      Bank:(Lb, C)  memory bank (tensor, same device)
    Ritorna: (Lq,) tensore con distanza minima per patch.
    """
    Lq = Q.shape[0]
    mins = []
    for s in range(0, Lq, block_b):
        q = Q[s:s+block_b]
        d = torch.cdist(q, Bank, p=2)  # (bs, Lb)
        mins.append(d.min(dim=1).values)
    return torch.cat(mins, dim=0)

# ======================================================
#                     MAIN
# ======================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

    # --------- DATA ---------
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=SEED,
        transform=None,
    )
    TRAIN_TAG = meta["train_tag"]
    print("[meta]", meta)
    if VIS_VALID_DATASET:
        show_dataset_images(val_set, batch_size=5, show_mask=True)

    print(f"Train GOOD (pos {meta['train_positions']}): {meta['counts']['train_good']}")
    print(f"Val   GOOD: {meta['counts']['val_good']}")
    print(f"Val  FAULT (pos {meta['val_fault_positions']}): {meta['counts']['val_fault']}")
    print(f"Val  TOT: {meta['counts']['val_total']}")

    # Loader: batch unico = 1 per train e validation (come richiesto)
    train_loader, val_loader = make_loaders(
        train_set, val_set,
        batch_size=BATCH_SIZE,
        device=device
    )

    # --------- MODEL ---------
    model, outputs = extract_backbone()
    model = model.to(device)

    # ==========================================================
    #        COSTRUZIONE/SALVATAGGIO PATCH MEMORY (TRAIN)
    # ==========================================================
    try:
        train_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[cache] FAPM train payload caricato.")
        bank2 = torch.from_numpy(train_payload["bank_l2"]).float()   # (Nb2, C2)
        bank3 = torch.from_numpy(train_payload["bank_l3"]).float()   # (Nb3, C3)
        fmap_shape = tuple(train_payload["shape"])
    except FileNotFoundError:
        print("[cache] Nessun pickle train: estraggo patch-bank layer-wise...")
        model.eval()
        feats2, feats3 = [], []

        with torch.inference_mode():
            for (x, _, _) in tqdm(train_loader, desc="| feature extraction | train | FAPM |"):
                x = x.to(device, non_blocking=True)
                outputs.clear()
                _ = model(x)
                l2, l3 = [t.detach() for t in outputs[:2]]  # (B,C,H,W) per layer
                outputs.clear()

                p2, p3, HW = feature_maps_to_patches(l2, l3)
                fmap_shape = HW
                feats2.append(p2.cpu().reshape(-1, p2.shape[-1]))  # (B*L, C2)
                feats3.append(p3.cpu().reshape(-1, p3.shape[-1]))  # (B*L, C3)

        bank2 = torch.cat(feats2, dim=0)  # (Npatch_tot, C2)
        bank3 = torch.cat(feats3, dim=0)  # (Npatch_tot, C3)
        del feats2, feats3

        # --------- CORESET ADATTIVO (patch-wise, per layer) ---------
        if CORESET_FRAC < 1.0:
            print(f"[coreset] start (frac={CORESET_FRAC})...")
            if CORESET_PROJ_D:
                X2 = maybe_random_project(bank2.numpy().astype(np.float32), CORESET_PROJ_D, seed=SEED)
                X3 = maybe_random_project(bank3.numpy().astype(np.float32), CORESET_PROJ_D, seed=SEED)
                X2t = torch.from_numpy(X2)
                X3t = torch.from_numpy(X3)
            else:
                X2t = bank2
                X3t = bank3

            m2 = max(1, int(CORESET_FRAC * X2t.shape[0]))
            m3 = max(1, int(CORESET_FRAC * X3t.shape[0]))
            idx2 = greedy_coreset_kcenter(X2t.float(), m=m2, seed=SEED)
            idx3 = greedy_coreset_kcenter(X3t.float(), m=m3, seed=SEED)
            bank2 = bank2[idx2]
            bank3 = bank3[idx3]
            print(f"[coreset] bank2 -> {bank2.shape[0]}  bank3 -> {bank3.shape[0]}")

        train_payload = {
            "version": 1,
            "cfg": {
                "layers": ["layer2", "layer3"],
                "img_size": int(IMG_SIZE),
                "coreset_frac": float(CORESET_FRAC),
                "coreset_proj_d": CORESET_PROJ_D if CORESET_PROJ_D is not None else -1,
                "train_tag": TRAIN_TAG,
            },
            "bank_l2": bank2.numpy().astype(np.float32),
            "bank_l3": bank3.numpy().astype(np.float32),
            "shape": (int(fmap_shape[0]), int(fmap_shape[1])),
        }
        save_split_pickle(train_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print(">> Train patch-memory salvato su pickle.")

    # portiamo i bank sul device per l'inferenza
    bank2 = bank2.to(device, non_blocking=True)
    bank3 = bank3.to(device, non_blocking=True)

    # ==========================================================
    #                 INFERENCE / VALIDATION
    # ==========================================================
    raw_score_maps = []
    img_scores = []
    gt_list = []

    model.eval()
    with torch.inference_mode():
        for (x, y, m) in tqdm(val_loader, desc="| inference | validation | FAPM |"):
            gt_list.extend(y.cpu().numpy())
            x = x.to(device, non_blocking=True)

            outputs.clear()
            _ = model(x)
            l2, l3 = [t.detach() for t in outputs[:2]]
            outputs.clear()

            # patch embeddings (B=1, L, C)
            p2, p3, HW = feature_maps_to_patches(l2, l3)
            H, W = HW
            B, L = p2.shape[0], p2.shape[1]

            # con batch_size=1: B=1 → prendo l'unico elemento
            q2 = p2[0].to(device)  # (L, C2)
            q3 = p3[0].to(device)  # (L, C3)

            d2 = top1_distance_per_patch(q2, bank2, block_b=8192)   # (L,)
            d3 = top1_distance_per_patch(q3, bank3, block_b=8192)   # (L,)

            d = 0.5 * (d2 + d3)                                     # (L,)

            score = d.view(H, W).unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
            score = F.interpolate(score, size=IMG_SIZE, mode='bilinear', align_corners=False)
            score = score.squeeze().detach().cpu().numpy().astype(np.float32)
            if GAUSSIAN_SIGMA > 0:
                score = gaussian_filter(score, sigma=GAUSSIAN_SIGMA)

            raw_score_maps.append(score)
            img_scores.append(float(score.max()))

            del l2, l3, p2, p3, x

    # --- metriche image-level ---
    gt_np = np.asarray(gt_list, dtype=np.int32)
    img_scores = np.asarray(img_scores, dtype=np.float32)
    fpr, tpr, thresholds = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0,1]).ravel()

    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    # Plot ROC rapido
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc_img:.3f}")
    ax.plot([0,1],[0,1],'k--',linewidth=1)
    ax.set_title("Image-level ROC"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    plt.tight_layout(); plt.show()

    # Visual (tua grid)
    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True, overlay_alpha=0.45
        )

    # --- valutazione pixel-level con la tua utility ---
    results = run_pixel_level_evaluation(
        score_map_list=list(raw_score_maps),   # heatmap RAW
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
