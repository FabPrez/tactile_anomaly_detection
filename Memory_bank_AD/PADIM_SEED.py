# padim.py
# PaDiM (Patch Distribution Modeling) - integrato con tue utility + seed separati
# - TEST_SEED: controlla split/val
# - TRAIN_SEED: controlla sottoinsieme GOOD usato nel training (good_fraction)
# - Supporto:
#   (1) main() singolo run con plotting
#   (2) run_single_experiment() batch singolo run (senza vis)
#   (3) run_all_fractions_for_current_piece()  -> 1 pezzo x tutte le frazioni
#   (4) run_all_seeds_and_fractions_for_current_piece() -> 1 pezzo x seed sweep x frazioni
#   (5) run_all_pieces_seeds_fractions() -> tutti i pezzi x seed sweep x frazioni
#   (6) run_all_pieces_and_fractions() -> tutti i pezzi x frazioni (seed fisso)  <<< NEW

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
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from scipy.ndimage import gaussian_filter


# ============================================================
# stampa Recall (x) quando Precision (y) = 0.900 nella PR
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
METHOD = "PADIM"
CODICE_PEZZO = "PZ1"

# Posizioni "good" per il TRAIN (feature bank)
TRAIN_POSITIONS = ["pos1"]

# Quanti GOOD per posizione spostare in VALIDATION (ed escludere dal TRAIN)
VAL_GOOD_PER_POS = 20

# Da quali posizioni prendere GOOD e FAULT per la VALIDATION
VAL_GOOD_SCOPE  = ["pos1"]     # "from_train" | "all_positions" | lista
VAL_FAULT_SCOPE = ["pos1"]     # "train_only" | "all" | lista

# Percentuale di GOOD (dopo il taglio per la val) da usare nel TRAIN
GOOD_FRACTION = 0.2

# Mappa pezzo → posizione da usare (stile InReaCh/FAPM)
PIECE_TO_POSITION = {
    "PZ1": "pos1",
    "PZ2": "pos5",
    "PZ3": "pos1",
    "PZ4": "pos1",
    "PZ5": "pos1",
}

# PaDiM (Patch Distribution Modeling)
PADIM_D   = 550                  # canali selezionati (<= C_total)
IMG_SIZE  = 224
GAUSSIAN_SIGMA = 4
RIDGE = 0.01                     # stabilizzazione cov

# seed separati
TEST_SEED  = 42   # controlla SOLO la scelta validation/test
TRAIN_SEED = 3   # controlla SOLO la scelta del sottoinsieme GOOD nel training

# sweep seed (come fai poi nei grafici seed0..seed9)
TRAIN_SEEDS_TO_RUN = list(range(10))   # [0..9]

# Visualizzazioni
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = False
# ------------------------------------------


# ----- util -----
def embedding_concat_nn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Allineamento PaDiM via nearest-neighbor + concat canali.
    x: (B, C1, H1, W1), y: (B, C2, H2, W2) con H1/W1 multipli.
    out: (B, C1+C2, H1, W1)
    """
    y_up = F.interpolate(y, size=(x.shape[-2], x.shape[-1]), mode='nearest')
    return torch.cat([x, y_up], dim=1)


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _payload_matches_cfg(train_payload: dict, train_tag: str) -> bool:
    """
    Evita di riusare pickle 'sporchi' se train_seed/test_seed non coincidono.
    """
    try:
        cfg = train_payload.get("cfg", {}) or {}
        ok = (
            int(cfg.get("test_seed", -1)) == int(TEST_SEED)
            and int(cfg.get("train_seed", -1)) == int(TRAIN_SEED)
            and str(cfg.get("train_tag", "")) == str(train_tag)
            and int(cfg.get("img_size", -1)) == int(IMG_SIZE)
        )
        return bool(ok)
    except Exception:
        return False


def main():
    # ======== DATASETS & LOADERS ========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ======== TRAINING STREAMING (2 PASS: mean -> cov) ========
    train_payload = None
    try:
        tmp = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        if _payload_matches_cfg(tmp, TRAIN_TAG):
            train_payload = tmp
            print(f"[cache] PaDiM train payload caricato ({METHOD}).")
        else:
            print("[cache] pickle trovato ma NON compatibile con i seed/tag correnti -> rebuild.")
            raise FileNotFoundError
    except FileNotFoundError:
        print("[cache] Nessun pickle train: avvio training streaming (mean -> cov).")

        rng = torch.Generator().manual_seed(1024)  # canali sempre uguali (repo-like)
        sel_idx = None
        d = None
        H = W = L = None

        # ---------- PASSO 1: stima MEDIA in streaming ----------
        N = 0
        sum_x = None  # torch tensor (d, L)

        with torch.inference_mode():
            for (x, _, _) in tqdm(train_loader, desc="| pass1 mean |"):
                _ = model(x.to(device, non_blocking=True))
                l1, l2, l3 = [t.cpu() for t in outputs[:3]]
                outputs.clear()

                emb_b = embedding_concat_nn(l1, l2)
                emb_b = embedding_concat_nn(emb_b, l3)          # (B, C_total, H, W)
                del l1, l2, l3

                if sel_idx is None:
                    C_total = emb_b.shape[1]
                    d = min(PADIM_D, C_total)
                    sel_idx = torch.randperm(C_total, generator=rng)[:d].tolist()

                emb_b = emb_b[:, sel_idx, :, :]               # (B, d, H, W)
                B = emb_b.shape[0]
                H, W = emb_b.shape[-2], emb_b.shape[-1]
                L = H * W

                E = emb_b.view(B, d, L).to(torch.float32)     # (B, d, L)

                if sum_x is None:
                    sum_x = E.sum(dim=0)                      # (d, L)
                else:
                    sum_x += E.sum(dim=0)
                N += B

                del emb_b, E, x
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        mean = (sum_x / float(N)).cpu().numpy().astype(np.float32)  # (d, L)
        del sum_x

        # ---------- PASSO 2: stima COVARIANZA in streaming + tiling su L ----------
        cov = np.zeros((d, d, L), dtype=np.float32)
        TILE  = 256

        with torch.inference_mode():
            for (x, _, _) in tqdm(train_loader, desc="| pass2 cov |"):
                _ = model(x.to(device, non_blocking=True))
                l1, l2, l3 = [t.cpu() for t in outputs[:3]]
                outputs.clear()

                emb_b = embedding_concat_nn(l1, l2)
                emb_b = embedding_concat_nn(emb_b, l3)        # (B, C_total, H, W)
                emb_b = emb_b[:, sel_idx, :, :]               # (B, d, H, W)
                B = emb_b.shape[0]
                E = emb_b.view(B, d, L).numpy().astype(np.float32)   # (B, d, L)
                del l1, l2, l3, emb_b

                for l0 in range(0, L, TILE):
                    l1_ = min(l0 + TILE, L)
                    diffs = E[:, :, l0:l1_] - mean[:, l0:l1_][None, :, :]   # (B, d, t)
                    cov[:, :, l0:l1_] += np.einsum('bdt,bkt->dkt', diffs, diffs, optimize=True).astype(np.float32)

                del E, x
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        cov /= float(max(1, N - 1))
        cov += (RIDGE * np.eye(d, dtype=np.float32))[:, :, None]

        train_payload = {
            "version": 1,
            "cfg": {
                "backbone": "wide_resnet50_2",
                "padim_d": int(d),
                "ridge": float(RIDGE),
                "img_size": int(IMG_SIZE),
                "seed_channels": 1024,
                "test_seed": int(TEST_SEED),
                "train_seed": int(TRAIN_SEED),
                "train_tag": TRAIN_TAG,
            },
            "mean": mean,
            "cov":  cov,
            "sel_idx": np.array(sel_idx, dtype=np.int64),
            "shape": (int(H), int(W)),
        }
        save_split_pickle(train_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print(">> Train feature bank salvato su pickle (mean+cov).")

    # ======== VALIDATION ========
    raw_score_maps = []
    gt_list = []

    with torch.inference_mode():
        for (x, y, m) in tqdm(val_loader, desc="| feature extraction | validation |"):
            gt_list.extend(y.cpu().numpy())

            _ = model(x.to(device, non_blocking=True))
            l1, l2, l3 = [t.cpu().detach() for t in outputs[:3]]
            outputs.clear()

            emb_t = embedding_concat_nn(l1, l2)
            emb_t = embedding_concat_nn(emb_t, l3)                # (B, Ctot, H, W)

            idx = torch.tensor(train_payload["sel_idx"], dtype=torch.long)
            Ht, Wt = train_payload["shape"]
            emb_t = torch.index_select(emb_t, 1, idx)             # (B, d, H, W)
            Bv, dv, Hc, Wc = emb_t.shape
            assert (Ht, Wt) == (Hc, Wc)

            emb_np_v = emb_t.view(Bv, dv, Hc * Wc).numpy().astype(np.float32)     # (B, d, L)
            mean_v   = train_payload["mean"]                                      # (d, L)
            cov_v    = train_payload["cov"]                                       # (d, d, L)

            Lloc = Hc * Wc
            TILE = 256
            dist2_LB = np.empty((Lloc, Bv), dtype=np.float32)

            for l0 in range(0, Lloc, TILE):
                l1_ = min(l0 + TILE, Lloc)
                # diffs: (t, Bv, d)
                diffs_t = np.transpose(emb_np_v[:, :, l0:l1_], (2, 0, 1)) - mean_v[:, l0:l1_].T[:, None, :]
                # cov: (t, d, d)
                cov_t = np.transpose(cov_v[:, :, l0:l1_], (2, 0, 1)).copy()
                eps = 1e-2
                cov_t += eps * np.eye(dv, dtype=np.float32)[None, :, :]

                cov_t_t = torch.from_numpy(cov_t)                 # (t,d,d)
                diffs_t_t = torch.from_numpy(diffs_t)             # (t,Bv,d)

                Lfac = torch.linalg.cholesky(cov_t_t)
                diffsT = diffs_t_t.transpose(1, 2).contiguous()   # (t,d,Bv)
                sol = torch.cholesky_solve(diffsT, Lfac)          # (t,d,Bv)
                dist2_tB = (diffsT * sol).sum(dim=1)              # (t,Bv)

                dist2_LB[l0:l1_, :] = dist2_tB.cpu().numpy().astype(np.float32)

            dist_arr = np.sqrt(dist2_LB.T).astype(np.float32).reshape(Bv, Hc, Wc)

            dist_t  = torch.from_numpy(dist_arr).unsqueeze(1)  # (B,1,H,W)
            score_b = F.interpolate(dist_t, size=IMG_SIZE, mode='bilinear',
                                    align_corners=False).squeeze(1).numpy()
            for i in range(score_b.shape[0]):
                score_b[i] = gaussian_filter(score_b[i], sigma=GAUSSIAN_SIGMA)

            raw_score_maps.extend([score_b[i] for i in range(score_b.shape[0])])

            del l1, l2, l3, emb_t, emb_np_v, dist_t, dist_arr, score_b, x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- normalizzazione SOLO per image score ----
    raw_score_maps = np.asarray(raw_score_maps, dtype=np.float32)   # (N, H, W)
    smax, smin = raw_score_maps.max(), raw_score_maps.min()
    scores_norm = (raw_score_maps - smin) / (smax - smin + 1e-12)

    img_scores_list = scores_norm.reshape(scores_norm.shape[0], -1).max(axis=1)
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

    # Plot ROC
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={auc_img:.3f}")
    ax[0].plot([0,1],[0,1],'k--',linewidth=1)
    ax[0].set_title("Image-level ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
    plt.tight_layout(); plt.show()

    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores_list, preds_img,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True, overlay_alpha=0.45
        )

    # ======== Valutazione pixel-level (tua utility) ========
    results = run_pixel_level_evaluation(
        score_map_list=list(raw_score_maps),    # heatmap RAW
        val_set=val_set,
        img_scores=img_scores_list,
        use_threshold="pro",
        fpr_limit=0.01,
        vis=False,
        vis_ds_or_loader=val_loader.dataset
    )

    # stampa recall quando precision=0.900 (curva PR)
    print_recall_when_precision_is(
        results,
        precision_target=0.900,
        tag=f"{METHOD}|{CODICE_PEZZO}|{TRAIN_TAG}|testSeed={TEST_SEED}|trainSeed={TRAIN_SEED}|gf={GOOD_FRACTION}"
    )

    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")


def run_single_experiment():
    """
    Esegue un esperimento completo usando le variabili globali:
        CODICE_PEZZO
        GOOD_FRACTION
        TEST_SEED
        TRAIN_SEED
    Ritorna:
        (image_auroc, pixel_auroc, pixel_auprc, pixel_aucpro)
    """
    _set_all_seeds(TRAIN_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    )
    TRAIN_TAG = meta["train_tag"]

    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=32, device=device)

    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model   = wide_resnet50_2(weights=weights).to(device)
    model.eval()

    outputs = []
    def hook(_m, _in, out): outputs.append(out)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    # ======== TRAIN (mean/cov) ========
    train_payload = None
    try:
        tmp = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        if _payload_matches_cfg(tmp, TRAIN_TAG):
            train_payload = tmp
            print(f"[cache] PaDiM train payload caricato ({METHOD}).")
        else:
            print("[cache] pickle trovato ma NON compatibile con i seed/tag correnti -> rebuild.")
            raise FileNotFoundError
    except FileNotFoundError:
        print("[cache] Nessun pickle train: avvio training streaming (mean -> cov).")

        rng = torch.Generator().manual_seed(1024)
        sel_idx = None
        d = None
        H = W = L = None

        # PASSO 1: mean
        N = 0
        sum_x = None

        with torch.inference_mode():
            for (x, _, _) in tqdm(train_loader, desc="| pass1 mean |"):
                _ = model(x.to(device, non_blocking=True))
                l1, l2, l3 = [t.cpu() for t in outputs[:3]]
                outputs.clear()

                emb_b = embedding_concat_nn(l1, l2)
                emb_b = embedding_concat_nn(emb_b, l3)
                del l1, l2, l3

                if sel_idx is None:
                    C_total = emb_b.shape[1]
                    d = min(PADIM_D, C_total)
                    sel_idx = torch.randperm(C_total, generator=rng)[:d].tolist()

                emb_b = emb_b[:, sel_idx, :, :]
                B = emb_b.shape[0]
                H, W = emb_b.shape[-2], emb_b.shape[-1]
                L = H * W

                E = emb_b.view(B, d, L).to(torch.float32)

                if sum_x is None:
                    sum_x = E.sum(dim=0)
                else:
                    sum_x += E.sum(dim=0)
                N += B

                del emb_b, E, x
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        mean = (sum_x / float(N)).cpu().numpy().astype(np.float32)
        del sum_x

        # PASSO 2: cov
        cov = np.zeros((d, d, L), dtype=np.float32)
        TILE  = 256

        with torch.inference_mode():
            for (x, _, _) in tqdm(train_loader, desc="| pass2 cov |"):
                _ = model(x.to(device, non_blocking=True))
                l1, l2, l3 = [t.cpu() for t in outputs[:3]]
                outputs.clear()

                emb_b = embedding_concat_nn(l1, l2)
                emb_b = embedding_concat_nn(emb_b, l3)
                emb_b = emb_b[:, sel_idx, :, :]
                B = emb_b.shape[0]
                E = emb_b.view(B, d, L).numpy().astype(np.float32)
                del l1, l2, l3, emb_b

                for l0 in range(0, L, TILE):
                    l1_ = min(l0 + TILE, L)
                    diffs = E[:, :, l0:l1_] - mean[:, l0:l1_][None, :, :]
                    cov[:, :, l0:l1_] += np.einsum('bdt,bkt->dkt', diffs, diffs, optimize=True).astype(np.float32)

                del E, x
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        cov /= float(max(1, N - 1))
        cov += (RIDGE * np.eye(d, dtype=np.float32))[:, :, None]

        train_payload = {
            "version": 1,
            "cfg": {
                "backbone": "wide_resnet50_2",
                "padim_d": int(d),
                "ridge": float(RIDGE),
                "img_size": int(IMG_SIZE),
                "seed_channels": 1024,
                "test_seed": int(TEST_SEED),
                "train_seed": int(TRAIN_SEED),
                "train_tag": TRAIN_TAG,
            },
            "mean": mean,
            "cov":  cov,
            "sel_idx": np.array(sel_idx, dtype=np.int64),
            "shape": (int(H), int(W)),
        }
        # Se vuoi cache anche nelle sweep seed: decommenta
        # save_split_pickle(train_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    # ======== VALIDATION ========
    raw_score_maps = []
    gt_list = []

    with torch.inference_mode():
        for (x, y, m) in tqdm(val_loader, desc="| feature extraction | validation |"):
            gt_list.extend(y.cpu().numpy())

            _ = model(x.to(device, non_blocking=True))
            l1, l2, l3 = [t.cpu().detach() for t in outputs[:3]]
            outputs.clear()

            emb_t = embedding_concat_nn(l1, l2)
            emb_t = embedding_concat_nn(emb_t, l3)

            idx = torch.tensor(train_payload["sel_idx"], dtype=torch.long)
            Ht, Wt = train_payload["shape"]
            emb_t = torch.index_select(emb_t, 1, idx)
            Bv, dv, Hc, Wc = emb_t.shape
            assert (Ht, Wt) == (Hc, Wc)

            emb_np_v = emb_t.view(Bv, dv, Hc * Wc).numpy().astype(np.float32)
            mean_v   = train_payload["mean"]
            cov_v    = train_payload["cov"]

            Lloc = Hc * Wc
            TILE = 256
            dist2_LB = np.empty((Lloc, Bv), dtype=np.float32)

            for l0 in range(0, Lloc, TILE):
                l1_ = min(l0 + TILE, Lloc)
                diffs_t = np.transpose(emb_np_v[:, :, l0:l1_], (2, 0, 1)) - mean_v[:, l0:l1_].T[:, None, :]
                cov_t = np.transpose(cov_v[:, :, l0:l1_], (2, 0, 1)).copy()
                eps = 1e-2
                cov_t += eps * np.eye(dv, dtype=np.float32)[None, :, :]

                cov_t_t = torch.from_numpy(cov_t)
                diffs_t_t = torch.from_numpy(diffs_t)

                Lfac = torch.linalg.cholesky(cov_t_t)
                diffsT = diffs_t_t.transpose(1, 2).contiguous()
                sol = torch.cholesky_solve(diffsT, Lfac)
                dist2_tB = (diffsT * sol).sum(dim=1)

                dist2_LB[l0:l1_, :] = dist2_tB.cpu().numpy().astype(np.float32)

            dist_arr = np.sqrt(dist2_LB.T).astype(np.float32).reshape(Bv, Hc, Wc)

            dist_t  = torch.from_numpy(dist_arr).unsqueeze(1)
            score_b = F.interpolate(dist_t, size=IMG_SIZE, mode='bilinear',
                                    align_corners=False).squeeze(1).numpy()
            for i in range(score_b.shape[0]):
                score_b[i] = gaussian_filter(score_b[i], sigma=GAUSSIAN_SIGMA)

            raw_score_maps.extend([score_b[i] for i in range(score_b.shape[0])])

            del l1, l2, l3, emb_t, emb_np_v, dist_t, dist_arr, score_b, x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    raw_score_maps = np.asarray(raw_score_maps, dtype=np.float32)
    smax, smin = raw_score_maps.max(), raw_score_maps.min()
    scores_norm = (raw_score_maps - smin) / (smax - smin + 1e-12)
    img_scores_list = scores_norm.reshape(scores_norm.shape[0], -1).max(axis=1)
    gt_np = np.asarray(gt_list, dtype=np.int32)

    # ==== IMAGE-LEVEL ====
    fpr, tpr, thresholds = roc_curve(gt_np, img_scores_list)
    auc_img = roc_auc_score(gt_np, img_scores_list)

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])
    preds_img = (img_scores_list >= best_thr).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(gt_np, preds_img, labels=[0, 1]).ravel()

    # ==== PIXEL-LEVEL ====
    results = run_pixel_level_evaluation(
        score_map_list=list(raw_score_maps),
        val_set=val_set,
        img_scores=img_scores_list,
        use_threshold="pro",
        fpr_limit=0.01,
        vis=False,
        vis_ds_or_loader=None
    )

    # stampa recall quando precision=0.900 (curva PR)
    print_recall_when_precision_is(
        results,
        precision_target=0.900,
        tag=f"{METHOD}|{CODICE_PEZZO}|{TRAIN_TAG}|testSeed={TEST_SEED}|trainSeed={TRAIN_SEED}|gf={GOOD_FRACTION}"
    )

    pixel_auroc   = float(results["curves"]["roc"]["auc"])
    pixel_auprc   = float(results["curves"]["pr"]["auprc"])
    pixel_auc_pro = float(results["curves"]["pro"]["auc"])

    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}, gf={GOOD_FRACTION}, train_seed={TRAIN_SEED}): {auc_img:.3f}")
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG} | gf={GOOD_FRACTION} | train_seed={TRAIN_SEED}")

    return float(auc_img), pixel_auroc, pixel_auprc, pixel_auc_pro


def run_all_fractions_for_current_piece():
    """
    Esegue più esperimenti variando GOOD_FRACTION per il pezzo corrente (CODICE_PEZZO),
    tenendo fisso TRAIN_SEED (globale).
    """
    global GOOD_FRACTION

    good_fracs = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00,
    ]

    img_list   = []
    pxroc_list = []
    pxpr_list  = []
    pxpro_list = []

    for gf in good_fracs:
        GOOD_FRACTION = gf
        print(f"\n=== PaDiM | PEZZO {CODICE_PEZZO}, TRAIN_SEED={TRAIN_SEED}, FRAZIONE GOOD = {GOOD_FRACTION} ===")
        auc_img, px_auroc, px_auprc, px_aucpro = run_single_experiment()

        img_list.append(auc_img)
        pxroc_list.append(px_auroc)
        pxpr_list.append(px_auprc)
        pxpro_list.append(px_aucpro)

    print(f"\n### RISULTATI PaDiM PER PEZZO {CODICE_PEZZO} (train_seed={TRAIN_SEED})")
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


def run_all_seeds_and_fractions_for_current_piece():
    """
    Esegue (TRAIN_SEED sweep) × (GOOD_FRACTION sweep) per il pezzo corrente.
    Output “seed-major”:
      results["pixel_auc_pro"][seed_idx][gf_idx]
    """
    global TRAIN_SEED, GOOD_FRACTION

    good_fracs = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95, 1.00,
    ]

    image_level_AUROC   = []
    pixel_level_AUROC   = []
    pixel_level_AUPRC   = []
    pixel_level_AUC_PRO = []

    for s in TRAIN_SEEDS_TO_RUN:
        TRAIN_SEED = s

        img_list   = []
        pxroc_list = []
        pxpr_list  = []
        pxpro_list = []

        for gf in good_fracs:
            GOOD_FRACTION = gf
            print(f"\n=== PaDiM | {CODICE_PEZZO} | train_seed={TRAIN_SEED} | gf={GOOD_FRACTION} ===")
            auc_img, px_auroc, px_auprc, px_aucpro = run_single_experiment()
            img_list.append(auc_img)
            pxroc_list.append(px_auroc)
            pxpr_list.append(px_auprc)
            pxpro_list.append(px_aucpro)

        image_level_AUROC.append(img_list)
        pixel_level_AUROC.append(pxroc_list)
        pixel_level_AUPRC.append(pxpr_list)
        pixel_level_AUC_PRO.append(pxpro_list)

    print(f"\n### RISULTATI PaDiM PER PEZZO {CODICE_PEZZO} (seeds={TRAIN_SEEDS_TO_RUN})")
    print("good_fractions      =", good_fracs)
    print("image_level_AUROC   =", image_level_AUROC)
    print("pixel_level_AUROC   =", pixel_level_AUROC)
    print("pixel_level_AUPRC   =", pixel_level_AUPRC)
    print("pixel_level_AUC_PRO =", pixel_level_AUC_PRO)

    return {
        "good_fractions": good_fracs,
        "train_seeds": list(TRAIN_SEEDS_TO_RUN),
        "image_auroc": image_level_AUROC,
        "pixel_auroc": pixel_level_AUROC,
        "pixel_auprc": pixel_level_AUPRC,
        "pixel_auc_pro": pixel_level_AUC_PRO,
    }


def run_all_pieces_seeds_fractions():
    """
    Esegue TUTTI i pezzi × (TRAIN_SEED sweep) × (GOOD_FRACTION sweep).
    """
    global CODICE_PEZZO, TRAIN_POSITIONS, VAL_GOOD_SCOPE, VAL_FAULT_SCOPE

    pieces = ["PZ1", "PZ2", "PZ3", "PZ4", "PZ5"]
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
        print(f"   PaDiM - RUNNING PIECE: {CODICE_PEZZO}")
        print(f"   POSITION:             {pos}")
        print(f"   TEST_SEED:            {TEST_SEED}")
        print(f"   TRAIN_SEEDS:          {TRAIN_SEEDS_TO_RUN}")
        print(f"============================")

        res = run_all_seeds_and_fractions_for_current_piece()
        all_results[pezzo] = res

    print("\n\n========================================")
    print("      PaDiM - RIEPILOGO TOTALE (seed×gf)")
    print("========================================\n")

    for pezzo, res in all_results.items():
        print(f"\n----- {pezzo} -----")
        print("train_seeds         =", res["train_seeds"])
        print("good_fractions      =", res["good_fractions"])
        print("image_level_AUROC   =", res["image_auroc"])
        print("pixel_level_AUROC   =", res["pixel_auroc"])
        print("pixel_level_AUPRC   =", res["pixel_auprc"])
        print("pixel_level_AUC_PRO =", res["pixel_auc_pro"])

    return all_results


# ============================
# NEW: come FAPM -> tutti i pezzi × tutte le frazioni (seed fisso)
# ============================
def run_all_pieces_and_fractions():
    """
    Esegue TUTTI i pezzi × TUTTE le frazioni, con TRAIN_SEED fisso (globale).
    Usa variabili GLOBALI sovrascritte ogni volta:
      - CODICE_PEZZO
      - TRAIN_POSITIONS, VAL_GOOD_SCOPE, VAL_FAULT_SCOPE
    """
    global CODICE_PEZZO, TRAIN_POSITIONS, VAL_GOOD_SCOPE, VAL_FAULT_SCOPE

    # scegli qui i pezzi che vuoi far girare
    pieces = ["PZ1", "PZ2", "PZ3", "PZ4", "PZ5"]
    # pieces = ["PZ4"]  # esempio

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
        print(f"   PaDiM - RUNNING PIECE: {CODICE_PEZZO}")
        print(f"   POSITION:             {pos}")
        print(f"   TEST_SEED:            {TEST_SEED}")
        print(f"   TRAIN_SEED (fixed):   {TRAIN_SEED}")
        print(f"============================")

        res = run_all_fractions_for_current_piece()
        all_results[pezzo] = res

    print("\n\n========================================")
    print("      PaDiM - RIEPILOGO TOTALE (fixed seed)")
    print("========================================\n")

    for pezzo, res in all_results.items():
        print(f"\n----- {pezzo} -----")
        print("good_fractions      =", res["good_fractions"])
        print("image_level_AUROC   =", res["image_auroc"])
        print("pixel_level_AUROC   =", res["pixel_auroc"])
        print("pixel_level_AUPRC   =", res["pixel_auprc"])
        print("pixel_level_AUC_PRO =", res["pixel_auc_pro"])

    return all_results


if __name__ == "__main__":
    # 1) SOLO 1 ESPERIMENTO “classico” con plotting/vis
    # main()

    # 2) SOLO 1 ESPERIMENTO “batch” (senza vis, ritorna metriche)
    # run_single_experiment()

    # 3) TUTTE LE FRAZIONI PER IL PEZZO CORRENTE (train_seed fisso)
    # run_all_fractions_for_current_piece()

    # 4) SEED sweep × GF sweep per il pezzo corrente
    # run_all_seeds_and_fractions_for_current_piece()

    # 5) TUTTI I PEZZI × (SEED sweep) × (GF sweep)
    # run_all_pieces_seeds_fractions()

    # 6) TUTTI I PEZZI × TUTTE LE FRAZIONI (seed fisso)  
    run_all_pieces_and_fractions()
