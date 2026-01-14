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


# ============================================================
# stampa Recall (x) quando Precision (y) = 0.900 nella PR
# 
# ============================================================
def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _find_x_at_y(x: np.ndarray, y: np.ndarray, y0: float):
    """
    Trova tutti gli x (interpolati) per cui la curva (x,y) incrocia y=y0.
    Se non incrocia, restituisce comunque il punto più vicino a y0.
    Ritorna:
      - xs_cross: lista di x dove y=y0
      - x_near, y_near: punto più vicino a y0
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return [], None, None

    diff = y - y0
    xs_cross = []

    # punti esatti
    exact_idx = np.where(diff == 0.0)[0]
    if exact_idx.size > 0:
        xs_cross.extend([float(x[i]) for i in exact_idx])

    # incroci tra punti consecutivi (cambio di segno)
    cross_idx = np.where(diff[:-1] * diff[1:] < 0.0)[0]
    for i in cross_idx:
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]
        if y2 == y1:
            continue
        t = (y0 - y1) / (y2 - y1)
        xs_cross.append(float(x1 + t * (x2 - x1)))

    # punto più vicino
    j = int(np.argmin(np.abs(diff)))
    x_near = float(x[j])
    y_near = float(y[j])

    # dedup + sort
    xs_cross = sorted(set([round(v, 12) for v in xs_cross]))
    return xs_cross, x_near, y_near


def print_recall_when_precision_is(results: dict, precision_target: float = 0.900, tag: str = ""):
    """
    Assumendo PR con:
      x = recall
      y = precision
    stampa recall @ precision=precision_target (anche più valori se la curva incrocia più volte).
    """
    pr = (results.get("curves", {}) or {}).get("pr", {}) or {}

    # prova chiavi più probabili
    recall = _first_not_none(
        pr.get("recall", None),
        pr.get("x", None),
        pr.get("rec", None),
    )
    precision = _first_not_none(
        pr.get("precision", None),
        pr.get("y", None),
        pr.get("prec", None),
    )

    if recall is None or precision is None:
        print(
            f"[PR]{'['+tag+']' if tag else ''} Non trovo gli array della curva PR in results['curves']['pr'] "
            f"(servono recall/precision oppure x/y). Chiavi disponibili: {list(pr.keys())}"
        )
        return

    recall = np.asarray(recall, dtype=np.float64).ravel()
    precision = np.asarray(precision, dtype=np.float64).ravel()

    xs, x_near, y_near = _find_x_at_y(recall, precision, precision_target)

    if xs:
        xs_str = ", ".join([f"{v:.6f}" for v in xs])
        print(f"[PR]{'['+tag+']' if tag else ''} recall @ precision={precision_target:.3f} -> {xs_str}")
    else:
        print(
            f"[PR]{'['+tag+']' if tag else ''} la curva NON incrocia precision={precision_target:.3f}. "
            f"Punto più vicino: recall={x_near:.6f} con precision={y_near:.6f}"
        )


# ----------------- CONFIG -----------------
METHOD = "SPADE"
CODICE_PEZZO = "PZ1"

# Posizioni "good" usate per il TRAIN (feature bank).
TRAIN_POSITIONS = ["pos1","pos2"]

# Quanti GOOD per posizione spostare nella VALIDATION (e quindi togliere dal TRAIN).
# Può essere:
#   - int  -> numero assoluto per ogni pos nello scope
#   - float fra 0 e 1 -> percentuale dei GOOD di quella pos
#   - dict, es: {"pos1": 10, "pos2": 0.2} (10 img per pos1, 20% per pos2)
VAL_GOOD_PER_POS = {
    "pos1":20,
    "pos2": 20
    
}

# Da quali posizioni prelevare i GOOD per la VALIDATION:
#   "from_train"     -> solo dalle pos di TRAIN_POSITIONS
#   "all_positions"  -> da tutte le pos del pezzo
#   ["pos1","pos3"]  -> lista custom
VAL_GOOD_SCOPE = ["pos1","pos2"]

# Da quali posizioni prendere le FAULT per la VALIDATION:
#   "train_only" | "all" | lista custom (es. ["pos1","pos2"])
VAL_FAULT_SCOPE = ["pos1","pos2"]

# Percentuale di GOOD (rimasti dopo aver tolto quelli per la val) da usare nel TRAIN.
# Può essere:
#   - float globale, es. 0.2  → 20% per tutte le pos di train
#   - dict per-posizione, es:
#       GOOD_FRACTION = {"pos1": 0.2, "pos2": 0.05}
#     (20% dei good di pos1, 5% dei good di pos2 dopo la rimozione per la val;
#      le pos non presenti nel dict usano 1.0 di default).
GOOD_FRACTION = {
    "pos1": 1.0,
    "pos2": 0.05
    
}

# Mappa pezzo → posizione (una sola per pezzo, come in InReaCh)
PIECE_TO_POSITION = {
    "PZ1": "pos1",
    "PZ2": "pos5",
    "PZ3": "pos1",
    "PZ4": "pos1",
    "PZ5": "pos1",
}

# Modello / dati
TOP_K    = 5
IMG_SIZE = 224

# >>> NEW: seed separati
TEST_SEED  = 42  # controlla *solo* la scelta delle immagini di validation/test
TRAIN_SEED = 9  # lo puoi cambiare tu per variare il sottoinsieme di GOOD usati per il training

# Visualizzazioni
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = False
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


# ---------- debug GOOD_FRACTION effettivo ----------
def debug_print_good_fraction_effective(meta):
    """
    Stampa la GOOD_FRACTION effettiva per posizione,
    cioè (good_train_after_fraction / good_total).
    """
    per_pos = meta.get("per_pos_counts", {})
    if not per_pos:
        print("\n[debug] GOOD_FRACTION effettivo: nessun dato per posizione.\n")
        return

    print("\n[debug] GOOD_FRACTION effettivo:")
    for pos, stats in per_pos.items():
        tot = stats.get("good_total", 0)
        train_final = stats.get("good_train_after_fraction",
                                stats.get("good_train", 0))

        if tot > 0:
            frac = train_final / tot
        else:
            frac = 0.0

        print(f"  - {pos}: {frac:.3f} ({frac*100:.1f}%)")
    print()


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======== DATASETS & LOADERS (presi da data_loader) ========
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=TEST_SEED,
        train_seed=TRAIN_SEED,
        transform=None,
        debug_print_val_paths=False,   # <<< accendi la stampa
    )
    TRAIN_TAG = meta["train_tag"]
    print("[meta]", meta)
    debug_print_good_fraction_effective(meta)

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

    # >>> NEW: stampa recall quando precision=0.900 (curva PR)
    print_recall_when_precision_is(results, precision_target=0.900, tag=f"{METHOD}|{CODICE_PEZZO}|{TRAIN_TAG}")

    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")


def run_single_experiment():
    """
    Esegue un esperimento completo SPADE usando le variabili globali.
    Ritorna:
        (image_auroc, pixel_auroc, pixel_auprc, pixel_aucpro)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======== DATASETS & LOADERS ========
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO,
        img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION,
        seed=TEST_SEED,
        train_seed=TRAIN_SEED,
        transform=None,
        debug_print_val_paths=False,   # <<< accendi la stampa
    )
    TRAIN_TAG = meta["train_tag"]

    if VIS_VALID_DATASET:
        show_dataset_images(val_set, batch_size=5, show_mask=True)

    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=32, device=device)

    # ====== MODELLO + HOOK ======
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
        # save_split_pickle(train_outputs, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

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

    # ====== IMAGE-LEVEL: KNN su avgpool ======
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

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0, 1]).ravel()

    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True,
            overlay_alpha=0.45
        )

    # ---- PIXEL LEVEL FEATURES ----
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

    # ---- Valutazione pixel-level (NO visual nelle sweep) ----
    results = run_pixel_level_evaluation(
        score_map_list=score_map_list,
        val_set=val_set,
        img_scores=img_scores,
        use_threshold="pro",
        fpr_limit=0.01,
        vis=False,
        vis_ds_or_loader=None
    )

    # >>> NEW: stampa recall quando precision=0.900 (curva PR)
    # tag più “sweep-friendly”: includo gf corrente
    print_recall_when_precision_is(results, precision_target=0.900, tag=f"{METHOD}|{CODICE_PEZZO}|gf={GOOD_FRACTION}")

    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")

    pixel_auroc   = float(results["curves"]["roc"]["auc"])
    pixel_auprc   = float(results["curves"]["pr"]["auprc"])
    pixel_auc_pro = float(results["curves"]["pro"]["auc"])

    return float(auc_img), pixel_auroc, pixel_auprc, pixel_auc_pro


# ============================================================
# ============   SWEEP SULLE FRAZIONI (UN PEZZO)   ===========
# ============================================================
def run_all_fractions_for_current_piece():
    """
    Esegue più esperimenti variando GOOD_FRACTION per il pezzo corrente (CODICE_PEZZO).
    Adatta GOOD_FRACTION al formato dict per-posizione:
        GOOD_FRACTION = {pos: frac}
    dove 'pos' è la (o le) posizioni in TRAIN_POSITIONS.
    """
    global GOOD_FRACTION

    # stesse frazioni usate per InReaCh (0.05 .. 1.0)
    good_fracs = [
        0.05, 0.10, 0.15, #0.20, 0.25,
        #0.30, 0.35, 0.40, 0.45, 0.50,
        #0.55, 0.60, 0.65, 0.70, 0.75,
        #0.80, 0.85, 0.90, 0.95, 1.00,
    ]

    img_list   = []
    pxroc_list = []
    pxpr_list  = []
    pxpro_list = []

    for gf in good_fracs:
        GOOD_FRACTION = {"pos1": 1.0, "pos2": gf}
        print(f"\n=== SPADE | {CODICE_PEZZO}, GOOD_FRACTION = {GOOD_FRACTION} ===")
       # run_single_experiment()

    #train_pos_list = list(TRAIN_POSITIONS)

    #for gf in good_fracs:
       # GOOD_FRACTION = {pos: gf for pos in train_pos_list}

        # print(f"\n=== PEZZO {CODICE_PEZZO}, FRAZIONE {gf} ===")
        auc_img, px_auroc, px_auprc, px_aucpro = run_single_experiment()

        img_list.append(auc_img)
        pxroc_list.append(px_auroc)
        pxpr_list.append(px_auprc)
        pxpro_list.append(px_aucpro)

    print("\n### RISULTATI PER PEZZO", CODICE_PEZZO)
    print("good_fractions      =", good_fracs)
    print("image_level_AUROC   =", img_list)
    print("pixel_level_AUROC   =", pxroc_list)
    print("pixel_level_AUPRC   =", pxpr_list)
    print("pixel_level_AUC_PRO =", pxpro_list)

    return {
        "good_fractions": good_fracs,
        "image_auroc": img_list,
        "pixel_auroc": pxroc_list,
        "pixel_auprc": pxpr_list,
        "pixel_auc_pro": pxpro_list,
    }


# ============================================================
# ========   TUTTI I PEZZI × TUTTE LE FRAZIONI   =============
# ============================================================
def run_all_pieces_and_fractions():
    """
    Esegue TUTTI i pezzi e TUTTE le frazioni.
    Usa variabili GLOBALI sovrascritte ogni volta:
      - CODICE_PEZZO
      - TRAIN_POSITIONS, VAL_GOOD_SCOPE, VAL_FAULT_SCOPE
      - GOOD_FRACTION (costruito come dict per-posizione)
    """
    global CODICE_PEZZO, TRAIN_POSITIONS, VAL_GOOD_SCOPE, VAL_FAULT_SCOPE, GOOD_FRACTION

    # pieces = ["PZ1", "PZ2", "PZ3", "PZ4", "PZ5"]
    # pieces = ["PZ2", "PZ3", "PZ4", "PZ5"]
    pieces = ["PZ1"]

    all_results = {}

    for pezzo in pieces:
        CODICE_PEZZO = pezzo

        if pezzo not in PIECE_TO_POSITION:
            raise ValueError(f"Nessuna posizione definita in PIECE_TO_POSITION per il pezzo {pezzo}")

        pos = PIECE_TO_POSITION[pezzo]

        TRAIN_POSITIONS = [pos]
        VAL_GOOD_SCOPE  = [pos]
        VAL_FAULT_SCOPE = [pos]

        print(f"\n\n============================")
        print(f"   RUNNING PIECE: {CODICE_PEZZO}")
        print(f"   POSITION:      {pos}")
        print(f"============================")

        res = run_all_fractions_for_current_piece()
        all_results[pezzo] = res

    print("\n\n========================================")
    print("           RIEPILOGO TOTALE (SPADE)")
    print("========================================\n")

    for pezzo, res in all_results.items():
        print(f"\n----- {pezzo} -----")
        print("good_fractions      =", res["good_fractions"])
        print("image_level_AUROC   =", res["image_auroc"])
        print("pixel_level_AUROC   =", res["pixel_auroc"])
        print("pixel_level_AUPRC   =", res["pixel_auprc"])
        print("pixel_level_AUC_PRO =", res["pixel_auc_pro"])

    return all_results


def entry_main():
    """
    Wrapper per scegliere facilmente cosa eseguire.
    Puoi usare questo invece di main() se vuoi solo le sweep.
    """
    # ESEGUI UN SOLO ESPERIMENTO (usa le globali correnti)
    #run_single_experiment()

    # TUTTE LE FRAZIONI PER UN SOLO PEZZO
    # CODICE_PEZZO = "PZ3"
    # pos = PIECE_TO_POSITION[CODICE_PEZZO]
    # TRAIN_POSITIONS[:] = [pos]
    # VAL_GOOD_SCOPE[:]  = [pos]
    # VAL_FAULT_SCOPE[:] = [pos]
    run_all_fractions_for_current_piece()

    # TUTTI I PEZZI × TUTTE LE FRAZIONI
    # run_all_pieces_and_fractions()


if __name__ == "__main__":
    # puoi scegliere se usare main() “vecchio” o entry_main()
    # main()
    entry_main()
