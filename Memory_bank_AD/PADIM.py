# padim.py
import os, random
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report
from view_utils import show_dataset_images, show_validation_grid_from_loader
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
from scipy.ndimage import gaussian_filter

# ----------------- CONFIG -----------------
METHOD = "PADIM"
CODICE_PEZZO = "PZ1"

# Posizioni "good" per il TRAIN (feature bank)
TRAIN_POSITIONS = ["pos1","pos2"]

# Quanti GOOD per posizione spostare in VALIDATION (ed escludere dal TRAIN)
VAL_GOOD_PER_POS = 20             # int oppure {"pos1": 10, "pos2": 30, ...}

# Da quali posizioni prendere GOOD e FAULT per la VALIDATION
VAL_GOOD_SCOPE  = ["pos1","pos2"]    # "from_train" | "all_positions" | lista ["pos1","pos2","pos3"]
VAL_FAULT_SCOPE = ["pos1","pos2"]    # "train_only" | "all" | lista

# Percentuale di GOOD (dopo il taglio per la val) da usare nel TRAIN
GOOD_FRACTION = 0.8              # 1.0=100%, 0.1=10%  ➜ nel tag: ...@p30

# PaDiM
PADIM_D   = 550                   # #canali da campionare (min(#canali, PADIM_D))
IMG_SIZE  = 224
SEED      = 42
GAUSSIAN_SIGMA = 4

# Visualizzazioni
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True
# ------------------------------------------


# ----- util -----
def embedding_concat(x, y):
    # concat spaziale dei livelli come in PaDiM
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)          # (B, C1*s*s, H2*W2)
    x = x.view(B, C1, -1, H2, W2)                                  # (B, C1, s*s, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2, device=x.device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)    # concat canali
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)  # torna a (B, C1+C2, H1, W1)
    return z


def main():
    # ======== DATASETS & LOADERS ========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    
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

    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=32, device=device)

    # ======== MODEL ========
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model   = wide_resnet50_2(weights=weights).to(device)
    model.eval()

    # Hook: come nell'originale
    outputs = []
    def hook(_m, _in, out): outputs.append(out)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    train_feats = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # ======== TRAIN FEATURE EXTRACTION ========
    # (cache opzionale commentata; resta compatibile con i tuoi pickle)
    train_emb_batches = []

    with torch.inference_mode():
        for (x, _, _) in tqdm(train_loader, desc="| feature extraction | train |"):
            _ = model(x.to(device, non_blocking=True))
            l1, l2, l3 = [t.cpu().detach() for t in outputs[:3]]
            outputs.clear()

            # concat spaziale per-BATCH (molto più leggero)
            emb_b = embedding_concat(l1, l2)
            emb_b = embedding_concat(emb_b, l3)      # (B, C_total, H, W)
            train_emb_batches.append(emb_b)

    # unisci solo adesso
    emb = torch.cat(train_emb_batches, dim=0)        # (N, C_total, H, W)
    del train_emb_batches, l1, l2, l3    # (N, C_total, H, W)

    # scelta canali deterministica (come reference: sample su C_total con seed=1024)
    B, C_total, H, W = emb.shape
    d  = min(PADIM_D, C_total)
    rng = torch.Generator().manual_seed(1024)
    sel_idx = torch.randperm(C_total, generator=rng)[:d]
    emb = torch.index_select(emb, 1, sel_idx)              # (N, d, H, W)

    # gaussian params per ogni location (H*W)
    emb = emb.view(B, d, H * W)                            # (N, d, L)
    mean = torch.mean(emb, dim=0).cpu().numpy()            # (d, L)
    cov  = np.zeros((d, d, H * W), dtype=np.float32)
    I    = np.eye(d, dtype=np.float32)
    emb_np = emb.cpu().numpy()
    for i in range(H * W):
        cov[:, :, i] = np.cov(emb_np[:, :, i], rowvar=False).astype(np.float32) + 0.01 * I

    train_payload = {
        "mean": mean,                              # (d, L)
        "cov":  cov,                               # (d, d, L)
        "sel_idx": sel_idx.cpu().numpy(),          # coerenza con la validazione
        "shape": (H, W),
    }
    # save_split_pickle(train_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    print(">> Train feature bank ready.")

    # ======== VALIDATION ========
    raw_score_maps = []   # raccolgo le mappe PRIMA della normalizzazione (come nel reference)
    img_scores_list = []  # verrà riempito DOPO normalizzazione globale
    gt_list, gt_mask_list = [], []

    with torch.inference_mode():
        for (x, y, m) in tqdm(val_loader, desc="| feature extraction | validation |"):
            gt_list.extend(y.cpu().numpy())
            gt_mask_list.extend(m.cpu().numpy())

            _ = model(x.to(device, non_blocking=True))
            # prendi le 3 feature del batch su CPU e pulisci
            l1, l2, l3 = [t.cpu().detach() for t in outputs[:3]]
            outputs.clear()
            print("i'm here just for debug1!")

            # concat spaziale su CPU
            emb_t = embedding_concat(l1, l2)
            emb_t = embedding_concat(emb_t, l3)                # (B, Ctot, H, W)
            print("i'm here just for debug2!")
            # stessa selezione canali del train
            idx = torch.tensor(train_payload["sel_idx"], dtype=torch.long)
            Ht, Wt = train_payload["shape"]
            emb_t = torch.index_select(emb_t, 1, idx)          # (B, d, H, W)
            Bv, dv, Hc, Wc = emb_t.shape
            assert (Ht, Wt) == (Hc, Wc)
            print("i'm here just for debug3!")

            # -> NumPy per Mahalanobis (per-pixel)
            emb_np_v = emb_t.view(Bv, dv, Hc * Wc).numpy()     # (B, d, L)
            mean_v   = train_payload["mean"]                   # (d, L)
            cov_v    = train_payload["cov"]                    # (d, d, L)
            print("i'm here just for debug4!")

           # ---- Mahalanobis per-pixel con Cholesky batched e tiling ----
            # Shapes: emb_np_v (Bv, d, L), mean_v (d, L), cov_v (d, d, L)
            L = Hc * Wc
            TILE = 256  # puoi provare 512 se hai più RAM/VRAM

            # prealloc per le distanze
            dist2_LB = np.empty((L, Bv), dtype=np.float32)

            # lavoriamo a tile di location per limitare la memoria
            for l0 in range(0, L, TILE):
                l1 = min(l0 + TILE, L)
                t = l1 - l0

                # diffs: (t, Bv, d)
                diffs_t = np.transpose(emb_np_v[:, :, l0:l1], (2, 0, 1)) - mean_v[:, l0:l1].T[:, None, :]  # (t,Bv,d)

                # covariances: (t, d, d)
                cov_t = np.transpose(cov_v[:, :, l0:l1], (2, 0, 1))  # (t,d,d)
                # piccola ridge extra per stabilità numerica
                eps = 1e-2
                cov_t = cov_t + eps * np.eye(dv, dtype=np.float32)[None, :, :]

                # -> torch per batched Cholesky/solve (CPU o GPU)
                cov_t_t = torch.from_numpy(cov_t)            # (t,d,d), float32
                diffs_t_t = torch.from_numpy(diffs_t)        # (t,Bv,d)

                # Cholesky (assume SPD grazie alla ridge)
                Lfac = torch.linalg.cholesky(cov_t_t)        # (t,d,d)

                # Risolvi C^{-1} * diff^T via cholesky_solve
                # diff^T: (t,d,Bv)
                diffsT = diffs_t_t.transpose(1, 2).contiguous()  # (t,d,Bv)
                sol = torch.cholesky_solve(diffsT, Lfac)         # (t,d,Bv) = inv(C) * diff^T

                # Mahalanobis^2 = sum(diff^T * (invC * diff^T), dim=d)
                dist2_tB = (diffsT * sol).sum(dim=1)             # (t,Bv)

                dist2_LB[l0:l1, :] = dist2_tB.cpu().numpy().astype(np.float32)

            # back to (Bv, Hc, Wc)
            dist_arr = np.sqrt(dist2_LB.T, dtype=np.float32).reshape(Bv, Hc, Wc)            # (B, H, W)

            # upsample + gaussian su CPU (come reference)
            dist_t  = torch.from_numpy(dist_arr).unsqueeze(1)  # (B,1,H,W)
            score_b = F.interpolate(dist_t, size=IMG_SIZE, mode='bilinear',
                                    align_corners=False).squeeze(1).numpy()
            # gaussian smoothing
            for i in range(score_b.shape[0]):
                score_b[i] = gaussian_filter(score_b[i], sigma=GAUSSIAN_SIGMA)
            print("i'm here just for debug6!")

            # ACCUMULO mappe grezze (niente min-max qui!)
            raw_score_maps.extend([score_b[i] for i in range(score_b.shape[0])])

            # cleanup
            del l1, l2, l3, emb_t, emb_np_v, dist_t, dist_arr, score_b
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- ----
    raw_score_maps = np.asarray(raw_score_maps, dtype=np.float32)   # (N, H, W)
    # normaliz ---
    smax, smin = raw_score_maps.max(), raw_score_maps.min()
    scores = (raw_score_maps - smin) / (smax - smin + 1e-12)        # (N, H, W)
    
    # image-level score: max spaziale dopo normalizzazione
    img_scores_list = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_np = np.asarray(gt_list, dtype=np.int32)

    # ROC AUC image-level
    fpr, tpr, thresholds = roc_curve(gt_np, img_scores_list)
    auc_img = roc_auc_score(gt_np, img_scores_list)
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")

    # Soglia (Youden) per classificazione image-level
    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])
    preds_img = (img_scores_list >= best_thr).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(gt_np, preds_img, labels=[0, 1]).ravel()
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    # Plot ROC (fix variabile)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={auc_img:.3f}")
    ax[0].plot([0,1],[0,1],'k--',linewidth=1)
    ax[0].set_title("Image-level ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
    plt.tight_layout(); plt.show()

    # Visualizzazione griglia su validation (opzionale)
    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores_list, preds_img,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True, overlay_alpha=0.45
        )

    # ---- (Opzionale) threshold pixel-level alla PaDiM (via PR/F1) ----
    # gt_mask_np = np.asarray(gt_mask_list, dtype=np.uint8)  # (N, H, W)
    # precision, recall, thr_pr = precision_recall_curve(gt_mask_np.flatten(), scores.flatten())
    # f1 = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(precision), where=(precision+recall)!=0)
    # thr_pixel = thr_pr[np.argmax(f1)] if len(thr_pr) > 0 else 0.5
    # print(f"[pixel-level] PR/F1 threshold (PaDiM-like): {thr_pixel:.4f}")

    # ======== Valutazione pixel-level (tua utility) ========
    results = run_pixel_level_evaluation(
        score_map_list=list(raw_score_maps),   
        val_set=val_set,
        img_scores=img_scores_list,
        use_threshold="pro",           # "roc" | "pr" | "pro"  (resto uguale)
        fpr_limit=0.01,
        vis=True,
        vis_ds_or_loader=val_loader.dataset
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")
    


if __name__ == "__main__":
    main()
