# SPADE.py
import torch, sys, platform
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import math

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import (
    roc_curve, accuracy_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score
)

# >>> NEW: per componenti connesse nelle GT
from scipy.ndimage import label as cc_label

# miei pacchetti
from data_loader import save_split_pickle, load_split_pickle, build_ad_datasets, make_loaders
from view_utils import show_dataset_images, show_validation_grid_from_loader, show_heatmaps_from_loader
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# ----------------- CONFIG -----------------
METHOD = "SPADE"
CODICE_PEZZO = "PZ3"

# Posizioni "good" usate per il TRAIN (feature bank).
TRAIN_POSITIONS = ["pos1"]

# Quanti GOOD per posizione spostare nella VALIDATION (e quindi togliere dal TRAIN).
# Può essere:
#   - int  -> numero assoluto per ogni pos nello scope
#   - float fra 0 e 1 -> percentuale dei GOOD di quella pos
#   - dict, es: {"pos1": 10, "pos2": 0.2} (10 img per pos1, 20% per pos2)
VAL_GOOD_PER_POS = 20

# Da quali posizioni prelevare i GOOD per la VALIDATION:
#   "from_train"     -> solo dalle pos di TRAIN_POSITIONS
#   "all_positions"  -> da tutte le pos del pezzo
#   ["pos1","pos3"]  -> lista custom
VAL_GOOD_SCOPE = ["pos1", "pos2"]

# Da quali posizioni prendere le FAULT per la VALIDATION:
#   "train_only" | "all" | lista custom (es. ["pos1","pos2"])
VAL_FAULT_SCOPE = ["pos1", "pos2"]

# Percentuale di GOOD (rimasti dopo aver tolto quelli per la val) da usare nel TRAIN.
# Può essere:
#   - float globale, es. 0.2  → 20% per tutte le pos di train
#   - dict per-posizione, es:
#       GOOD_FRACTION = {"pos1": 0.2, "pos2": 0.05}
#     (20% dei good di pos1, 5% dei good di pos2 dopo la rimozione per la val;
#      le pos non presenti nel dict usano 1.0 di default).
GOOD_FRACTION = {
    "pos1": 0.2,   # 20% pos1
    "pos2": 0.05,  # 5% pos2
}

# Modello / dati
TOP_K    = 5
IMG_SIZE = 224
SEED     = 42

# Visualizzazioni
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True
GAUSSIAN_SIGMA = 4
# ------------------------------------------


# ---------- util ----------
def calc_dist_matrix(x, y):
    """Euclidean distance matrix tra righe di x (n,d) e y (m,d) -> (n,m)."""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


def l2norm(x, dim=1, eps=1e-6):
    """Normalizzazione L2 lungo la dimensione 'dim' (cosine-like distance)."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@torch.no_grad()
def topk_cdist_streaming(X, Y, k=7, block_x=1024, block_y=4096, device=torch.device('cuda'), use_amp=True):
    """
    X: (N_test, D)  Y: (N_train, D)
    Restituisce: topk_values (N_test,k), topk_indexes (N_test,k)
    """
    X = X.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)

    is_cuda = (isinstance(device, torch.device) and device.type == 'cuda') or (str(device) == 'cuda')
    all_topv, all_topi = [], []

    for i in range(0, X.size(0), block_x):
        xi = X[i:i+block_x]  # (bx, D)
        vals_row, inds_row = None, None

        if use_amp and is_cuda:
            amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            amp_ctx = torch.cuda.amp.autocast(enabled=False)

        with amp_ctx:
            for j in range(0, Y.size(0), block_y):
                yj = Y[j:j+block_y]              # (by, D)
                d = torch.cdist(xi, yj)          # (bx, by)

                v, idx = torch.topk(d, k=min(k, d.size(1)), dim=1, largest=False)
                idx = idx + j                    # shift indici locali -> globali

                if vals_row is None:
                    vals_row, inds_row = v, idx
                else:
                    vals_row = torch.cat([vals_row, v], dim=1)
                    inds_row = torch.cat([inds_row, idx], dim=1)
                    vals_row, new_idx = torch.topk(vals_row, k=k, dim=1, largest=False)
                    inds_row = inds_row.gather(1, new_idx)

        all_topv.append(vals_row.float().cpu())
        all_topi.append(inds_row.cpu())

    topk_values = torch.cat(all_topv, dim=0)
    topk_indexes = torch.cat(all_topi, dim=0)
    return topk_values, topk_indexes


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device found:", device)

    # ======== DATASETS & LOADERS (presi da data_loader) ========
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,   # ora può essere int/float/dict
        good_fraction=GOOD_FRACTION,         # ora può essere float o dict per-posizione
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

    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=32, device=device)

    # modello + hook
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model   = wide_resnet50_2(weights=weights).to(device)
    model.eval()

    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    test_outputs  = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

    # ====== TRAIN FEATURES (cache SOLO TRAIN) ======
    try:
        train_outputs = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[cache] Train features caricate da pickle.")
    except FileNotFoundError:
        print("[cache] Nessun pickle train: estraggo feature...")
        for x, y, m in tqdm(train_loader, desc='| feature extraction | train | custom |'):
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                _ = model(x)
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.detach().cpu())
            outputs = []
        for k in train_outputs:
            train_outputs[k] = torch.cat(train_outputs[k], dim=0)
        save_split_pickle(train_outputs, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    # ====== VAL FEATURES ======
    gt_list = []
    for x, y, m in tqdm(val_loader, desc='| feature extraction | validation | custom |'):
        gt_list.extend(y.cpu().numpy())
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            _ = model(x)
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.detach().cpu())
        outputs = []
    for k in test_outputs:
        test_outputs[k] = torch.cat(test_outputs[k], dim=0)

    gt_np = np.asarray(gt_list, dtype=np.int32)
    for k in test_outputs:
        assert test_outputs[k].shape[0] == len(gt_np), f"Mismatch batch su {k}"

    # ====== IMAGE-LEVEL: KNN su avgpool (streaming, no N×M) ======
    X = torch.flatten(test_outputs['avgpool'], 1).to(torch.float32)
    Y = torch.flatten(train_outputs['avgpool'], 1).to(torch.float32)

    topk_values, topk_indexes = topk_cdist_streaming(
        X, Y, k=TOP_K,
        block_x=1024,
        block_y=4096,
        device=device,
        use_amp=True
    )

    img_scores = topk_values.mean(dim=1).cpu().numpy()

    fpr, tpr, thresholds = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])

    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0, 1]).ravel()
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(gt_list, img_scores):.3f}")
    ax[0].plot([0,1],[0,1],'k--',linewidth=1)
    ax[0].set_title("Image-level ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
    plt.tight_layout(); plt.show()

    print(f"[check] len(val_loader.dataset) = {len(val_loader.dataset)}")
    print(f"[check] len(scores)             = {len(img_scores)}")

    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True, overlay_alpha=0.45
        )

    # ---- PIXEL LEVEL FEATURES --------------------------
    score_map_list = []
    for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % CODICE_PEZZO):
        per_layer_maps = []
        for layer_name in ['layer1', 'layer2', 'layer3']:
            K = topk_indexes.shape[1]

            topk_feat = train_outputs[layer_name][topk_indexes[t_idx]].to(device)   # (K,C,H,W)
            test_feat = test_outputs[layer_name][t_idx:t_idx + 1].to(device)        # (1,C,H,W)

            topk_feat = l2norm(topk_feat, dim=1)
            test_feat = l2norm(test_feat, dim=1)

            K_, C, H, W = topk_feat.shape

            gallery = topk_feat.permute(0, 2, 3, 1).reshape(K_*H*W, C).contiguous()
            query   = test_feat.permute(0, 2, 3, 1).reshape(H*W, C).contiguous()

            B = 20000
            mins = []
            for s in range(0, gallery.shape[0], B):
                d = torch.cdist(gallery[s:s+B], query)     # (B, H*W)
                mins.append(d.min(dim=0).values)
            dist_min = torch.stack(mins, dim=0).min(dim=0).values  # (H*W,)

            score_map = dist_min.view(1, 1, H, W)
            score_map = F.interpolate(score_map, size=IMG_SIZE, mode='bilinear', align_corners=False)
            per_layer_maps.append(score_map.cpu())

        score_map = torch.mean(torch.cat(per_layer_maps, dim=0), dim=0)  # (1,1,224,224)
        score_map = score_map.squeeze().numpy()
        if GAUSSIAN_SIGMA > 0:
            score_map = gaussian_filter(score_map, sigma=GAUSSIAN_SIGMA)
        score_map_list.append(score_map)

    # ---- Valutazione & visualizzazione ----
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
