# fapm_official_adapted_cosine_coreset_norm.py
# FAPM (Fast Adaptive Patch Memory) — versione allineata a repo/paper
# - Backbone: Wide-ResNet-50-2, layer2 & layer3
# - Metriche: cosine (L2-norm feature; score = 1 - cos)
# - Coreset: adattivo patch-wise (seed "difficili" + k-center)
# - Fusione layer-wise: normalizzazione robusta (mediana/IQR) e media
#
# Dipendenze: torch, torchvision, numpy, scipy, sklearn, tqdm, matplotlib
# Utility Valerio: data_loader, view_utils, ad_analysis (come SPADE/PADIM)

import os, math, random
from collections import OrderedDict
from typing import Tuple, List, Dict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter

# --- tue utility condivise ---
from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from view_utils import show_dataset_images, show_validation_grid_from_loader
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

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
GOOD_FRACTION   = 1.0

# FAPM: 2 layer (l2, l3), banca patch per layer
IMG_SIZE        = 224
SEED            = 42
GAUSSIAN_SIGMA  = 4

# Coreset adattivo (patch-wise)
CORESET_FRAC    = 0.02       # 0.02 = tieni ~2% delle patch
CORESET_SEED_FR = 0.10       # percentuale patch "difficili" usate come seed-pool prima del k-center
CORESET_PROJ_D  = 128        # None per disattivare; altrimenti SRP a D dimensioni

# Normalizzazione per-layer (robusta) sui punteggi di distanza
NORM_SAMPLE_IMGS = 24        # quante immagini di train campionare per stimare mediana/IQR (se disponibili)

# Visual
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True

# Batch unico per train e validation (richiesta tua)
BATCH_SIZE = 1

# ======================================================
#                 HELPER
# ======================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def extract_backbone():
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model = wide_resnet50_2(weights=weights)
    model.eval()
    outputs = []
    def hook(_m, _in, out): outputs.append(out)
    # FAPM usa due scale → layer2 & layer3 (ultimo blocco)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    return model, outputs

def feature_maps_to_patches(fm2: torch.Tensor, fm3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int,int]]:
    """
    Converte feature map (B,C,H,W) in patch embeddings (B, H*W, C) per layer.
    - riallinea spatialmente l3 -> l2
    - L2 norm per canale (cosine-ready)
    """
    fm3u = F.interpolate(fm3, size=fm2.shape[-2:], mode='nearest')
    fm2  = l2norm(fm2, dim=1)
    fm3u = l2norm(fm3u, dim=1)

    B, C2, H, W = fm2.shape
    C3 = fm3u.shape[1]
    L  = H * W

    p2 = fm2.permute(0, 2, 3, 1).reshape(B, L, C2).contiguous()
    p3 = fm3u.permute(0, 2, 3, 1).reshape(B, L, C3).contiguous()
    return p2, p3, (H, W)

def cosine_top1_distance(Q: torch.Tensor, Bank: torch.Tensor, block_b: int = 8192) -> torch.Tensor:
    """
    Score per patch con metrica cosine (feature già L2-normalizzate):
      score = 1 - max_cosine_similarity
    Q:    (Lq, C)
    Bank: (Lb, C)
    return: (Lq,)
    """
    Lq = Q.shape[0]
    out = []
    for s in range(0, Lq, block_b):
        q = Q[s:s+block_b]                 # (bs, C)
        # cos = q @ Bank^T  (dato che q e Bank sono unit-norm per canale)
        cos = q @ Bank.t()                 # (bs, Lb)
        maxcos, _ = cos.max(dim=1)
        out.append(1.0 - maxcos)           # distanza = 1 - similarità
    return torch.cat(out, dim=0)

def greedy_kcenter(X: torch.Tensor, m: int, seed: int = 0) -> np.ndarray:
    """
    K-center "greedy" standard su L2.
    X: (N, D) CPU float32
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

def adaptive_coreset_from_difficult(X: torch.Tensor,
                                    frac_keep: float,
                                    frac_seed: float,
                                    proj_d: int | None,
                                    seed: int = 0) -> np.ndarray:
    """
    Coreset adattivo:
      1) Proiezione opzionale (Sparse Random Projection) per efficienza
      2) Stima centro globale e punteggio "difficoltà" = ||x - mu||_2
      3) Prendi top-q% "difficili" come seed-pool
      4) Esegui k-center sul seed-pool per selezionare m = frac_keep * N
    """
    N = X.shape[0]
    if frac_keep >= 1.0 or frac_keep <= 0.0 or N == 0:
        return np.arange(N, dtype=np.int64)

    # Proiezione SRP opzionale
    X_use = X
    if proj_d is not None and proj_d > 0 and proj_d < X.shape[1]:
        srp = SparseRandomProjection(n_components=proj_d, random_state=seed)
        X_use_np = X.cpu().numpy().astype(np.float32)
        X_use_np = srp.fit_transform(X_use_np).astype(np.float32)
        X_use = torch.from_numpy(X_use_np)

    # Centro globale & difficulty
    mu = X_use.mean(dim=0, keepdim=True)
    diff = torch.cdist(X_use, mu, p=2).squeeze(1)  # (N,)

    # Seed pool dalle top "difficili"
    k_seed = max(1, int(frac_seed * N))
    seed_idx = torch.topk(diff, k=k_seed, largest=True).indices.cpu().numpy()

    # k-center sul seed-pool
    X_seed = X[seed_idx]  # importante: k-center nello spazio originale (meno distorsione)
    m = max(1, int(frac_keep * N))
    sub_sel = greedy_kcenter(X_seed.float(), m=m, seed=seed)

    return seed_idx[sub_sel]  # indici rispetto a X originale

def robust_norm_stats(values: np.ndarray) -> Tuple[float, float]:
    """
    Ritorna (mediana, IQR) con guardia su IQR minimale.
    """
    med = float(np.median(values))
    q25 = float(np.percentile(values, 25.0))
    q75 = float(np.percentile(values, 75.0))
    iqr = max(1e-6, (q75 - q25))
    return med, iqr

def apply_robust_norm(values: np.ndarray, med: float, iqr: float) -> np.ndarray:
    return (values - med) / iqr

# ======================================================
#                     MAIN
# ======================================================
def main():
    set_seed(SEED)
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

    # Loader batch=1 (train/val)
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
    need_build = True
    try:
        train_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[cache] FAPM train payload caricato.")
        bank2 = torch.from_numpy(train_payload["bank_l2"]).float()
        bank3 = torch.from_numpy(train_payload["bank_l3"]).float()
        fmap_shape = tuple(train_payload["shape"])
        # stats per-layer per normalizzazione
        norm_stats = train_payload.get("norm_stats", None)
        if norm_stats is None or not all(k in norm_stats for k in ("l2_med","l2_iqr","l3_med","l3_iqr")):
            need_build = True  # ricostruisco per avere stats
        else:
            l2_med = float(norm_stats["l2_med"]); l2_iqr = float(norm_stats["l2_iqr"])
            l3_med = float(norm_stats["l3_med"]); l3_iqr = float(norm_stats["l3_iqr"])
            need_build = False
    except FileNotFoundError:
        need_build = True

    if need_build:
        print("[cache] Nessun pickle train valido: estraggo patch-bank layer-wise + coreset adattivo...")
        model.eval()
        feats2, feats3 = [], []

        with torch.inference_mode():
            for (x, _, _) in tqdm(train_loader, desc="| feature extraction | train | FAPM |"):
                x = x.to(device, non_blocking=True)
                outputs.clear()
                _ = model(x)
                l2, l3 = [t.detach() for t in outputs[:2]]  # (B,C,H,W)
                outputs.clear()

                p2, p3, HW = feature_maps_to_patches(l2, l3)
                fmap_shape = HW
                feats2.append(p2.cpu().reshape(-1, p2.shape[-1]))  # (B*L, C2)
                feats3.append(p3.cpu().reshape(-1, p3.shape[-1]))  # (B*L, C3)

        bank2_full = torch.cat(feats2, dim=0)  # (Npatch_tot, C2) L2-normalized
        bank3_full = torch.cat(feats3, dim=0)  # (Npatch_tot, C3) L2-normalized
        del feats2, feats3

        # --------- CORESET ADATTIVO (patch-wise, per layer) ---------
        if CORESET_FRAC < 1.0:
            print(f"[coreset/adaptive] frac_keep={CORESET_FRAC}, seed_pool={CORESET_SEED_FR}, proj_d={CORESET_PROJ_D}")
            idx2 = adaptive_coreset_from_difficult(
                bank2_full.float().cpu(), frac_keep=CORESET_FRAC, frac_seed=CORESET_SEED_FR,
                proj_d=CORESET_PROJ_D, seed=SEED
            )
            idx3 = adaptive_coreset_from_difficult(
                bank3_full.float().cpu(), frac_keep=CORESET_FRAC, frac_seed=CORESET_SEED_FR,
                proj_d=CORESET_PROJ_D, seed=SEED
            )
            bank2 = bank2_full[idx2]
            bank3 = bank3_full[idx3]
        else:
            bank2 = bank2_full
            bank3 = bank3_full

        # --------- Stima statistiche di normalizzazione per-layer ---------
        print("[norm] stimo mediana/IQR dei punteggi per layer su un campione di train...")
        # portiamo i bank sul device per calcolare distro punteggi
        bank2_d = bank2.to(device, non_blocking=True)
        bank3_d = bank3.to(device, non_blocking=True)

        samp_scores_l2 = []
        samp_scores_l3 = []

        model.eval()
        with torch.inference_mode():
            n_seen = 0
            for (x, _, _) in train_loader:
                x = x.to(device, non_blocking=True)
                outputs.clear()
                _ = model(x)
                l2, l3 = [t.detach() for t in outputs[:2]]
                outputs.clear()

                p2, p3, HW = feature_maps_to_patches(l2, l3)
                q2 = p2[0].to(device)  # (L, C2)
                q3 = p3[0].to(device)  # (L, C3)

                # cosine distance (1 - cos)
                d2 = cosine_top1_distance(q2, bank2_d, block_b=8192).detach().cpu().numpy()
                d3 = cosine_top1_distance(q3, bank3_d, block_b=8192).detach().cpu().numpy()

                # raccogli un campione (per non esplodere in RAM)
                L = d2.shape[0]
                if L > 1024:
                    sel = np.linspace(0, L-1, 1024, dtype=np.int64)
                    samp_scores_l2.append(d2[sel])
                    samp_scores_l3.append(d3[sel])
                else:
                    samp_scores_l2.append(d2)
                    samp_scores_l3.append(d3)

                n_seen += 1
                if n_seen >= NORM_SAMPLE_IMGS:
                    break

        samp2 = np.concatenate(samp_scores_l2, axis=0) if len(samp_scores_l2) else np.array([0.0], dtype=np.float32)
        samp3 = np.concatenate(samp_scores_l3, axis=0) if len(samp_scores_l3) else np.array([0.0], dtype=np.float32)

        l2_med, l2_iqr = robust_norm_stats(samp2)
        l3_med, l3_iqr = robust_norm_stats(samp3)
        norm_stats = {"l2_med": l2_med, "l2_iqr": l2_iqr, "l3_med": l3_med, "l3_iqr": l3_iqr}
        print(f"[norm] layer2: med={l2_med:.5f} iqr={l2_iqr:.5f} | layer3: med={l3_med:.5f} iqr={l3_iqr:.5f}")

        # --------- Salva payload ----------
        train_payload = {
            "version": 2,
            "cfg": {
                "layers": ["layer2", "layer3"],
                "img_size": int(IMG_SIZE),
                "coreset_frac": float(CORESET_FRAC),
                "coreset_seed_fr": float(CORESET_SEED_FR),
                "coreset_proj_d": CORESET_PROJ_D if CORESET_PROJ_D is not None else -1,
                "train_tag": TRAIN_TAG,
                "metric": "cosine_1_minus",
                "norm_strategy": "robust_median_iqr",
                "norm_sample_imgs": int(NORM_SAMPLE_IMGS),
            },
            "bank_l2": bank2.cpu().numpy().astype(np.float32),
            "bank_l3": bank3.cpu().numpy().astype(np.float32),
            "shape": (int(fmap_shape[0]), int(fmap_shape[1])),
            "norm_stats": norm_stats
        }
        save_split_pickle(train_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print(">> Train patch-memory + norm_stats salvati su pickle.")
    else:
        print("[norm] uso norm_stats dal pickle...")
        norm_stats = train_payload["norm_stats"]
        l2_med = float(norm_stats["l2_med"]); l2_iqr = float(norm_stats["l2_iqr"])
        l3_med = float(norm_stats["l3_med"]); l3_iqr = float(norm_stats["l3_iqr"])

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

            # patch embeddings (B=1, L, C) già L2-normalizzate
            p2, p3, HW = feature_maps_to_patches(l2, l3)
            H, W = HW
            q2 = p2[0].to(device)  # (L, C2)
            q3 = p3[0].to(device)  # (L, C3)

            # cosine distance (1 - cos)
            d2 = cosine_top1_distance(q2, bank2, block_b=8192).detach().cpu().numpy().astype(np.float32)  # (L,)
            d3 = cosine_top1_distance(q3, bank3, block_b=8192).detach().cpu().numpy().astype(np.float32)  # (L,)

            # normalizzazione robusta per-layer
            d2n = apply_robust_norm(d2, l2_med, l2_iqr)
            d3n = apply_robust_norm(d3, l3_med, l3_iqr)

            # fusione layer-wise (media dopo normalizzazione)
            d = 0.5 * (d2n + d3n)  # (L,)

            score = torch.from_numpy(d.reshape(H, W)).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
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

    # Plot ROC
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
        score_map_list=list(raw_score_maps),   # heatmap RAW (già smooth)
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
