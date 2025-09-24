# ===== Pixel-level toolkit (riusabile) =======================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import label as cc_label
from view_utils import show_heatmaps_from_loader

from pro_curve_util import compute_pro, trapezoid

# Se le hai già definite nel tuo file, rimuovi queste "import" ridondanti:
# from scipy.ndimage import label as cc_label

def build_gt_arrays(val_set):
    """Ritorna:
        - gt_pix: vettore flatten (N_tot_pixel,)
        - gt_mask_list: lista di mask 2D (H,W) per PRO
    """
    gt_pix = []
    gt_mask_list = []
    # pixel-level GT flatten
    loader_masks = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    for _, _, m in loader_masks:  # m: (B,H,W)
        gt_pix.append(m.numpy().reshape(m.size(0), -1))
    gt_pix = np.concatenate(gt_pix, axis=0).ravel().astype(np.uint8)

    # per-region GT (una mask per immagine)
    for _, _, m in DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0):
        gt_mask_list.append(m.squeeze(0).numpy().astype(np.uint8))

    return gt_pix, gt_mask_list


def compute_pixel_curves(score_map_list, gt_pix, gt_mask_list, fpr_limit=0.3, num_thrs=200):
    """Calcola curve ROC/PR (pixel-level) e PRO + AUC."""
    # Pred pixel flatten (score “grezze”, più alto = più anomalo)
    pred_pix = np.concatenate([sm.reshape(-1) for sm in score_map_list], axis=0).astype(np.float32)
    assert pred_pix.shape[0] == gt_pix.shape[0], "Pixels mismatch: pred vs gt"

    # ROC / AUROC
    fpr_pix, tpr_pix, thr_roc = roc_curve(gt_pix, pred_pix)
    auc_pix = roc_auc_score(gt_pix, pred_pix)

    # PR / AUPRC
    prec, rec, thr_pr = precision_recall_curve(gt_pix, pred_pix)
    auprc_pix = average_precision_score(gt_pix, pred_pix)

    # PRO
    fpr_pro, pro_vals, thr_vals, auc_pro = compute_pro_curve(
        score_map_list, gt_mask_list, fpr_limit=fpr_limit
    )

    return {
        "roc":  {"fpr": fpr_pix, "tpr": tpr_pix, "thr": thr_roc, "auc": auc_pix},
        "pr":   {"prec": prec, "rec": rec, "thr": thr_pr, "auprc": auprc_pix},
        "pro":  {"fpr": fpr_pro, "pro": pro_vals, "thr": thr_vals, "auc": auc_pro, "limit": fpr_limit},
        "pred_pix": pred_pix,  # utile per debug/altre analisi
    }


def select_thresholds(curves, mode_pro="max_pro"):
    """Seleziona soglie:
       - ROC: Youden (max TPR-FPR)
       - PR:  max F1
       - PRO: max PRO con FPR <= limite
    """
    # ROC → Youden
    fpr = curves["roc"]["fpr"]; tpr = curves["roc"]["tpr"]; thr_roc = curves["roc"]["thr"]
    J = tpr - fpr
    best_idx_roc = int(np.argmax(J))
    best_thr_roc = float(thr_roc[best_idx_roc])

    # PR → max F1 (note: thr_pr ha len = len(prec)-1)
    prec = curves["pr"]["prec"]; rec = curves["pr"]["rec"]; thr_pr = curves["pr"]["thr"]
    f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx_pr = int(np.argmax(f1_vals))
    best_thr_pr = float(thr_pr[best_idx_pr])

    # PRO → max PRO con FPR <= limite
    fpr_pro = curves["pro"]["fpr"]; pro_vals = curves["pro"]["pro"]; thr_vals = curves["pro"]["thr"]
    fpr_limit = curves["pro"]["limit"]
    best_thr_pro, idx_pro, fpr_at_best, pro_at_best = select_threshold_from_pro(
        fpr_pro, pro_vals, thr_vals, fpr_limit=fpr_limit, mode=mode_pro
    )

    return {
        "roc": {"thr": best_thr_roc, "idx": best_idx_roc},
        "pr":  {"thr": best_thr_pr,  "idx": best_idx_pr, "f1": float(f1_vals[best_idx_pr])},
        "pro": {"thr": best_thr_pro, "idx": idx_pro, "fpr": fpr_at_best, "pro": pro_at_best},
    }


def make_masks(score_map_list, thrs):
    """Genera maschere binarie per ogni criterio (liste di (H,W))."""
    masks_roc = [(sm >= thrs["roc"]["thr"]).astype(np.uint8) for sm in score_map_list]
    masks_pr  = [(sm >= thrs["pr"]["thr" ]).astype(np.uint8) for sm in score_map_list]
    masks_pro = [(sm >= thrs["pro"]["thr"]).astype(np.uint8) for sm in score_map_list]
    return {"roc": masks_roc, "pr": masks_pr, "pro": masks_pro}


def eval_masks(masks_dict, val_set):
    """Valuta Acc/P/R/F1 per ciascun set di maschere (usa eval_pixel_metrics del tuo codice)."""
    out = {}
    for k, masks in masks_dict.items():
        acc, p, r, f1 = eval_pixel_metrics(masks, val_set)
        out[k] = {"acc": acc, "prec": p, "rec": r, "f1": f1}
    return out


def plot_pixel_curves(curves, title_suffix=""):
    """Plot ROC, PR e PRO; ritorna le figure per eventuale salvataggio."""
    figs = {}

    # ROC
    fig1, ax1 = plt.subplots(1,1,figsize=(5,4))
    ax1.plot(curves["roc"]["fpr"], curves["roc"]["tpr"], label=f"AUROC={curves['roc']['auc']:.3f}")
    ax1.plot([0,1],[0,1],'k--',lw=1)
    ax1.set(xlabel="FPR", ylabel="TPR", title=f"Pixel ROC {title_suffix}")
    ax1.legend(loc="lower right"); fig1.tight_layout()
    figs["roc"] = fig1

    # PR
    fig2, ax2 = plt.subplots(1,1,figsize=(5,4))
    ax2.plot(curves["pr"]["rec"], curves["pr"]["prec"], label=f"AUPRC={curves['pr']['auprc']:.3f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title=f"Pixel PR {title_suffix}")
    ax2.legend(loc="lower left"); fig2.tight_layout()
    figs["pr"] = fig2

    # PRO
    fig3, ax3 = plt.subplots(1,1,figsize=(5,4))
    ax3.plot(curves["pro"]["fpr"], curves["pro"]["pro"], label=f"AUC-PRO@0.3={curves['pro']['auc']:.3f}")
    ax3.axvline(curves["pro"]["limit"], ls='--', lw=1)
    ax3.set(xlabel="FPR", ylabel="PRO", title=f"PRO curve {title_suffix}")
    ax3.legend(loc="lower right"); fig3.tight_layout()
    figs["pro"] = fig3

    return figs


def visualize_heatmaps(ds_or_loader, score_maps, img_scores, per_page=6, cols=3,
                       normalize_each=False, overlay_alpha=0.45, cmap="jet",
                       title_fmt="idx {i} | label {g} | score {s:.3f}"):
    """Wrapper sottile sulla tua show_heatmaps_from_loader (per coerenza firma)."""
    show_heatmaps_from_loader(
        ds_or_loader=ds_or_loader,
        score_maps=score_maps,
        scores=img_scores,
        per_page=per_page,
        cols=cols,
        normalize_each=normalize_each,
        overlay_alpha=overlay_alpha,
        cmap=cmap,
        title_fmt=title_fmt,
    )
    
def select_thresholds(curves):
    """Seleziona soglie:
       - ROC: Youden (max TPR-FPR)
       - PR:  max F1
       - PRO: max PRO con FPR <= fpr_limit
    """
    # ROC → Youden
    fpr = curves["roc"]["fpr"]; tpr = curves["roc"]["tpr"]; thr_roc = curves["roc"]["thr"]
    J = tpr - fpr
    best_idx_roc = int(np.argmax(J))
    best_thr_roc = float(thr_roc[best_idx_roc])

    # PR → max F1 (nota: thr_pr ha len = len(prec)-1)
    prec = curves["pr"]["prec"]; rec = curves["pr"]["rec"]; thr_pr = curves["pr"]["thr"]
    f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx_pr = int(np.argmax(f1_vals))
    best_thr_pr = float(thr_pr[best_idx_pr])

    # PRO → max PRO con FPR <= limite
    fpr_pro = curves["pro"]["fpr"]; pro_vals = curves["pro"]["pro"]; thr_vals = curves["pro"]["thr"]
    fpr_limit = curves["pro"]["limit"]
    best_thr_pro, idx_pro, fpr_at_best, pro_at_best = select_threshold_from_pro(
        fpr_pro, pro_vals, thr_vals, fpr_limit=fpr_limit, mode="max_pro"
    )

    return {
        "roc": {"thr": best_thr_roc, "idx": best_idx_roc},
        "pr":  {"thr": best_thr_pr,  "idx": best_idx_pr, "f1": float(f1_vals[best_idx_pr])},
        "pro": {"thr": best_thr_pro, "idx": idx_pro, "fpr": fpr_at_best, "pro": pro_at_best},
    }

def run_pixel_level_evaluation(score_map_list, val_set, img_scores,
                               use_threshold="pro",     # "roc" | "pr" | "pro"
                               fpr_limit=0.3,          # default 0.3
                               vis=True, vis_ds_or_loader=None):
    """
    Pipeline completa pixel-level:
      - GT pixel & per-regione
      - curve ROC/PR/PRO (AUC/AUPRC/AUC-PRO@fpr_limit)
      - soglie: ROC(Youden), PR(max F1), PRO(max PRO @ FPR≤fpr_limit)
      - maschere + metriche
      - visualizza CURVE e MASCHERE secondo `use_threshold`
    """
    use_threshold = str(use_threshold).lower()
    if use_threshold not in {"roc", "pr", "pro"}:
        raise ValueError("use_threshold deve essere 'roc', 'pr' o 'pro'")

    # 1) GT
    gt_pix, gt_mask_list = build_gt_arrays(val_set)

    # 2) curve
    curves = compute_pixel_curves(score_map_list, gt_pix, gt_mask_list,
                                  fpr_limit=fpr_limit, num_thrs=200)

    # 3) soglie (PRO = max PRO @ FPR≤fpr_limit)
    thrs = select_thresholds(curves)

    # 4) maschere + metriche
    masks = make_masks(score_map_list, thrs)
    metrics = eval_masks(masks, val_set)

    # selezione per visual/riporto
    maps_to_show = masks[use_threshold]
    selected_thr = thrs[use_threshold]["thr"]
    selected_metrics = metrics[use_threshold]

    # 5) visual
    figs = None
    if vis:
        figs = plot_pixel_curves(curves, title_suffix=f"(FPR≤{0.3}, {use_threshold.upper()})")
        visualize_heatmaps(
            ds_or_loader=(vis_ds_or_loader if vis_ds_or_loader is not None else val_set),
            score_maps=maps_to_show,        # binarie con la soglia scelta
            img_scores=img_scores,          # i tuoi image-level scores
            per_page=6, cols=3,
            normalize_each=False, overlay_alpha=0.45, cmap="jet"
        )

    return {
        "curves": curves,            # ROC/PR/PRO + AUC
        "thresholds": thrs,          # tutte: roc/pr/pro
        "masks": masks,              # maschere per ciascuna soglia
        "metrics": metrics,          # metriche aggregate
        "selected": {
            "key": use_threshold,
            "threshold": selected_thr,
            "masks": maps_to_show,
            "metrics": selected_metrics,
        },
        "figs": figs,
    }
# ============================================================================

def compute_pro_curve(score_map_list, gt_mask_list, fpr_limit=0.3):
    # 1) curva (FPR, PRO)
    fpr_arr, pro_arr = compute_pro(
        anomaly_maps=score_map_list,
        ground_truth_maps=gt_mask_list
    )

    # 2) AUC-PRO limitata e normalizzata
    au_pro = trapezoid(fpr_arr, pro_arr, x_max=0.3)
    auc_pro_norm = au_pro / max(0.3, 1e-12)

    # 3) ricostruzione soglie coerenti coi punti della curva
    scores_flat = np.array(score_map_list, dtype=np.float32).ravel()
    if scores_flat.size == 0:
        thr_vals = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        sort_idx = np.argsort(scores_flat)[::-1]              # ordina desc
        scores_sorted = scores_flat[sort_idx]
        keep_mask = np.append(np.diff(scores_sorted) != 0, True)
        thr_core = scores_sorted[keep_mask]
        if thr_core.size == 0:
            c = float(scores_sorted[0])
            thr_vals = np.array([c+1e-12, c, c-1e-12], dtype=np.float32)
        else:
            hi = float(thr_core[0]) + 1e-12
            lo = float(thr_core[-1]) - 1e-12
            thr_vals = np.concatenate(([hi], thr_core.astype(np.float32), [lo]))

    # 4) allineamento lunghezze
    if thr_vals.shape[0] < fpr_arr.shape[0]:
        thr_vals = np.pad(thr_vals, (0, fpr_arr.shape[0]-thr_vals.shape[0]), mode='edge')
    elif thr_vals.shape[0] > fpr_arr.shape[0]:
        thr_vals = thr_vals[:fpr_arr.shape[0]]

    return fpr_arr, pro_arr, thr_vals, auc_pro_norm



def _compute_fpr_and_region_overlaps(pred_bin_list, gt_list):
    """
    pred_bin_list: lista di array binari (H,W) predetti (0/1)
    gt_list:      lista di array binari (H,W) GT (0/1)
    Ritorna:
      fpr_scalar: FP / (FP+TN) aggregato su tutte le immagini
      overlaps:   lista di overlap per ciascuna regione connessa di GT (|P∩G|/|G|)
    """
    assert len(pred_bin_list) == len(gt_list)
    FP = 0
    TN = 0
    overlaps = []

    for pred, gt in zip(pred_bin_list, gt_list):
        # FPR globale sugli sfondi (gt==0)
        bg = (gt == 0)
        FP += np.logical_and(pred == 1, bg).sum()
        TN += np.logical_and(pred == 0, bg).sum()

        # PRO: per-region overlap sulle componenti connesse delle regioni GT
        if (gt > 0).any():
            lbl, ncomp = cc_label(gt.astype(np.uint8))
            for k in range(1, ncomp + 1):
                region = (lbl == k)
                denom = region.sum()
                if denom > 0:
                    num = np.logical_and(pred == 1, region).sum()
                    overlaps.append(num / float(denom))

    fpr = FP / float(FP + TN + 1e-12)
    return fpr, overlaps


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

def get_val_image_by_global_idx(val_loader, global_idx):
    """
    Ritorna l'immagine (CHW in [0,1]) del global_idx percorrendo val_loader in ordine.
    Richiede shuffle=False e stesse transform usate per le feature.
    """
    seen = 0
    for x, _ in val_loader:
        b = x.size(0)
        if seen + b > global_idx:
            return x[global_idx - seen].cpu().numpy()
        seen += b
    raise IndexError(f"indices out of range: {global_idx}, tot={seen}")


def select_threshold_from_pro(fpr, pro, thr, fpr_limit=0.3, mode="max_pro"):
    """
    Seleziona la soglia τ a partire dalla curva PRO:
      - fpr: array di FPR(τ) (pixel-level) per ciascuna soglia
      - pro: array di PRO(τ) media per-regione
      - thr: array delle soglie τ corrispondenti
    Vincolo: FPR(τ) <= fpr_limit.
    mode:
      - "max_pro": massimizza PRO sotto il vincolo FPR
      - "closest_to_limit": sceglie la soglia col FPR più vicino (da sinistra) al limite
      - "tradeoff": massimizza (PRO - FPR) sotto il vincolo

    Ritorna: (best_thr, best_idx, fpr_at_best, pro_at_best)
    """
    fpr = np.asarray(fpr, dtype=float)
    pro = np.asarray(pro, dtype=float)
    thr = np.asarray(thr, dtype=float)

    assert fpr.shape == pro.shape == thr.shape, "fpr/pro/thr devono avere stessa shape"
    if fpr.size == 0:
        raise ValueError("Curve PRO vuota.")

    # Considera solo i punti che rispettano il vincolo
    valid = np.where(fpr <= fpr_limit)[0]

    # Se nessun punto rispetta il vincolo: prendi il punto a FPR minimo (più vicino al limite da destra)
    if valid.size == 0:
        i = int(np.argmin(fpr))
        return float(thr[i]), i, float(fpr[i]), float(pro[i])

    if mode == "max_pro":
        # Massimizza PRO tra i validi; se c'è parità su PRO, scegli FPR più basso
        i_local = valid[np.argmax(pro[valid])]
        # tie-break opzionale
        ties = valid[np.isclose(pro[valid], pro[i_local])]
        if ties.size > 1:
            i_local = ties[np.argmin(fpr[ties])]
        i = int(i_local)

    elif mode == "closest_to_limit":
        # Scegli il punto con FPR più vicino al limite, restando sotto (da sinistra)
        i = int(valid[np.argmin(np.abs(fpr[valid] - fpr_limit))])

    elif mode == "tradeoff":
        # Massimizza (PRO - FPR) sotto il vincolo (semplice compromesso)
        score = pro[valid] - fpr[valid]
        i = int(valid[np.argmax(score)])
    else:
        raise ValueError("mode non riconosciuto: usa 'max_pro', 'closest_to_limit' o 'tradeoff'.")

    return float(thr[i]), i, float(fpr[i]), float(pro[i])



def eval_pixel_metrics(masks_bin_list, gt_dataset):
    # flatten pred
    pred_flat = np.concatenate([m.reshape(-1) for m in masks_bin_list], axis=0).astype(np.uint8)
    # flatten gt (già lo hai come gt_pix, ma ricalcoliamo per completezza)
    gt_flat = []
    loader_masks = DataLoader(gt_dataset, batch_size=32, shuffle=False, num_workers=0)
    for _, _, m in loader_masks:                      # m: (B,H,W) uint8 {0,1}
        gt_flat.append(m.numpy().reshape(m.size(0), -1))
    gt_flat = np.concatenate(gt_flat, axis=0).ravel().astype(np.uint8)

    assert gt_flat.shape[0] == pred_flat.shape[0], "Mismatch gt vs pred pixel."
    acc = accuracy_score(gt_flat, pred_flat)
    p   = precision_score(gt_flat, pred_flat, zero_division=0)
    r   = recall_score(gt_flat, pred_flat,    zero_division=0)
    f1  = f1_score(gt_flat, pred_flat,        zero_division=0)
    return acc, p, r, f1


def print_pixel_report(results, title=None):
    """
    Stampa un riepilogo compatto di:
      - AUROC / AUPRC / AUC-PRO (pixel-level)
      - soglie scelte (ROC/PR/PRO)
      - metriche pixel-level (Acc/Prec/Rec/F1) per ciascuna soglia
    """
    curves   = results["curves"]
    thrs     = results["thresholds"]
    metrics  = results["metrics"]
    pro_lim  = curves["pro"]["limit"]

    if title:
        print(f"\n===== {title} =====")

    # --- AUCs (pixel-level) ---
    auroc   = curves["roc"]["auc"]
    auprc   = curves["pr"]["auprc"]
    auc_pro = curves["pro"]["auc"]
    print(f"[AUCs | pixel-level]  AUROC={auroc:.4f}   AUPRC={auprc:.4f}   AUC-PRO@{pro_lim:.1f}={auc_pro:.4f}")

    # --- soglie scelte ---
    print("Soglie scelte:")
    print(f"  - ROC (Youden):    τ={thrs['roc']['thr']:.6f}   (idx={thrs['roc']['idx']})")
    print(f"  - PR  (max F1):    τ={thrs['pr']['thr']:.6f}    (idx={thrs['pr']['idx']}, F1*={thrs['pr']['f1']:.4f})")
    print(f"  - PRO (max PRO≤{pro_lim:.1f}): τ={thrs['pro']['thr']:.6f}  (idx={thrs['pro']['idx']}, FPR={thrs['pro']['fpr']:.4f}, PRO={thrs['pro']['pro']:.4f})")

    # --- metriche per ciascuna soglia ---
    def _fmt(m): return f"Acc={m['acc']:.4f}  Prec={m['prec']:.4f}  Rec={m['rec']:.4f}  F1={m['f1']:.4f}"
    print("Metriche pixel-level (con le soglie sopra):")
    print(f"  - ROC: {_fmt(metrics['roc'])}")
    print(f"  - PR : {_fmt(metrics['pr'])}")
    print(f"  - PRO: {_fmt(metrics['pro'])}")