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
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from scipy.ndimage import gaussian_filter

# ----------------- CONFIG -----------------
METHOD = "PADIM"
CODICE_PEZZO = "PZ3"

# Posizioni "good" per il TRAIN (feature bank)
TRAIN_POSITIONS = ["pos2"]

# Quanti GOOD per posizione spostare in VALIDATION (ed escludere dal TRAIN)
VAL_GOOD_PER_POS = 0

# Da quali posizioni prendere GOOD e FAULT per la VALIDATION
VAL_GOOD_SCOPE  = ["pos2"]     # "from_train" | "all_positions" | lista
VAL_FAULT_SCOPE = ["pos2"]     # "train_only" | "all" | lista

# Percentuale di GOOD (dopo il taglio per la val) da usare nel TRAIN
GOOD_FRACTION = 1.0

# PaDiM
PADIM_D   = 550                  # canali selezionati (<= C_total)
IMG_SIZE  = 224
SEED      = 42
GAUSSIAN_SIGMA = 4
RIDGE = 0.01                     # stabilizzazione cov

# Visualizzazioni
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True
# ------------------------------------------


# ----- util -----
def embedding_concat_nn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Allineamento PaDiM via nearest-neighbor (replica a blocchi s×s) + concat canali.
    Equivalente concettualmente a unfold/fold a blocchi, ma senza allocazioni grandi.
    x: (B, C1, H1, W1), y: (B, C2, H2, W2), con H1/H2 intero.
    out: (B, C1+C2, H1, W1)
    """
    y_up = F.interpolate(y, size=(x.shape[-2], x.shape[-1]), mode='nearest')
    return torch.cat([x, y_up], dim=1)


class PaDiMTileModel:
    """
    PaDiM per un singolo tile: salva mean/cov per ogni posizione, predice heatmap Mahalanobis.
    """
    def __init__(self, padim_d=550, img_size=224, ridge=0.01, gaussian_sigma=4, device="cpu"):
        self.padim_d = padim_d
        self.img_size = img_size
        self.ridge = ridge
        self.gaussian_sigma = gaussian_sigma
        self.device = torch.device(device)
        self.mean = None
        self.cov = None
        self.sel_idx = None
        self.H = None
        self.W = None

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        self.model = wide_resnet50_2(weights=weights).to(self.device).eval()
        self.outputs = []
        def hook(_m, _in, out): self.outputs.append(out)
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def _extract_embedding(self, img_np):
        img_t = torch.tensor(img_np, dtype=torch.float32)
        if img_t.max() > 1.0:
            img_t = img_t / 255.0
        img_t = img_t.permute(2,0,1).unsqueeze(0).to(self.device)
        self.outputs.clear()
        with torch.no_grad():
            _ = self.model(img_t)
        l1, l2, l3 = [t.cpu() for t in self.outputs[:3]]
        emb = embedding_concat_nn(l1, l2)
        emb = embedding_concat_nn(emb, l3)  # (1, Ctot, H, W)
        C_total = emb.shape[1]
        if self.sel_idx is None:
            d = min(self.padim_d, C_total)
            rng = torch.Generator().manual_seed(1024)
            self.sel_idx = torch.randperm(C_total, generator=rng)[:d].tolist()
        emb = emb[:, self.sel_idx, :, :]  # (1, d, H, W)
        B, d, H, W = emb.shape
        self.H, self.W = H, W
        return emb.view(B, d, H*W).squeeze(0)  # (d, L)

    def fit(self, tiles):
        # tiles: lista di immagini (H,W,C)
        feats = []
        for tile in tiles:
            emb = self._extract_embedding(tile)  # (d, L)
            feats.append(emb)
        X = torch.stack(feats, dim=0)  # (N, d, L)
        N, d, L = X.shape
        mean = X.mean(dim=0)  # (d, L)
        cov = torch.zeros((d, d, L), dtype=torch.float32)
        for l in range(L):
            diffs = X[:,:,l] - mean[:,l][None,:]  # (N, d)
            cov[:,:,l] = (diffs.t() @ diffs) / max(1, N-1)
            cov[:,:,l] += self.ridge * torch.eye(d)
        self.mean = mean.numpy().astype(np.float32)
        self.cov = cov.numpy().astype(np.float32)

    def predict(self, tile):
        emb = self._extract_embedding(tile)  # (d, L)
        mean = self.mean
        cov = self.cov
        d, L = emb.shape
        dist2 = np.zeros((L,), dtype=np.float32)
        TILE = 256
        for l0 in range(0, L, TILE):
            l1_ = min(l0 + TILE, L)
            t = l1_ - l0
            diffs = emb[:, l0:l1_].T - mean[:, l0:l1_].T  # (t, d)
            cov_t = np.transpose(cov[:, :, l0:l1_], (2, 0, 1)).copy()  # (t, d, d)
            eps = 1e-2
            cov_t += eps * np.eye(d, dtype=np.float32)[None, :, :]
            cov_t_t = torch.from_numpy(cov_t)
            diffs_t = torch.from_numpy(diffs)
            Lfac = torch.linalg.cholesky(cov_t_t)
            diffsT = diffs_t.transpose(0,1).unsqueeze(2)  # (d, t, 1)
            sol = torch.cholesky_solve(diffsT, Lfac)
            dist2_t = (diffsT.squeeze(2) * sol.squeeze(2)).sum(dim=0)
            dist2[l0:l1_] = dist2_t.cpu().numpy().astype(np.float32)
        dist_arr = np.sqrt(dist2).reshape(self.H, self.W)
        # upsample + gaussian
        dist_t = torch.from_numpy(dist_arr).unsqueeze(0).unsqueeze(0)
        score_map = F.interpolate(dist_t, size=self.img_size, mode='bilinear', align_corners=False).squeeze().numpy()
        if self.gaussian_sigma > 0:
            score_map = gaussian_filter(score_map, sigma=self.gaussian_sigma)
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        return score_map


def main():
    # ======== DATASETS & LOADERS ========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # force cpu

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

    # ======== TRAINING STREAMING (2 PASS: mean -> cov) ========
    # Prova a caricare la cache (se esiste salto il train)
    try:
        train_payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print(f"[cache] PaDiM train payload caricato ({METHOD}).")
        mean    = train_payload["mean"]          # (d, L)
        cov     = train_payload["cov"]           # (d, d, L)
        sel_idx = train_payload["sel_idx"]       # (d,)
        H, W    = train_payload["shape"]         # feature map size
        L       = H * W
    except FileNotFoundError:
        print("[cache] Nessun pickle train: avvio training streaming (mean -> cov).")

        # --- selezione canali: verrà inizializzata al primo batch ---
        rng = torch.Generator().manual_seed(1024)
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

                # allineamento e concat (come tuo reference)
                emb_b = embedding_concat_nn(l1, l2)
                emb_b = embedding_concat_nn(emb_b, l3)          # (B, C_total, H, W)
                del l1, l2, l3

                # inizializza sel_idx al primo batch
                if sel_idx is None:
                    C_total = emb_b.shape[1]
                    d = min(PADIM_D, C_total)
                    sel_idx = torch.randperm(C_total, generator=rng)[:d].tolist()

                # selezione canali SUBITO (riduce memoria)
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
        RIDGE = 0.01
        TILE  = 256  # puoi aumentare/diminuire in base alla RAM

        with torch.inference_mode():
            for (x, _, _) in tqdm(train_loader, desc="| pass2 cov |"):
                _ = model(x.to(device, non_blocking=True))
                l1, l2, l3 = [t.cpu() for t in outputs[:3]]
                outputs.clear()

                emb_b = embedding_concat_nn(l1, l2)
                emb_b = embedding_concat_nn(emb_b, l3)          # (B, C_total, H, W)
                emb_b = emb_b[:, sel_idx, :, :]               # (B, d, H, W)
                B = emb_b.shape[0]
                E = emb_b.view(B, d, L).numpy().astype(np.float32)   # (B, d, L)
                del l1, l2, l3, emb_b

                for l0 in range(0, L, TILE):
                    l1_ = min(l0 + TILE, L)
                    t = l1_ - l0

                    diffs = E[:, :, l0:l1_] - mean[:, l0:l1_][None, :, :]   # (B, d, t)
                    # somma degli outer products su B → (d, d, t)
                    cov[:, :, l0:l1_] += np.einsum('bdt,bkt->dkt', diffs, diffs, optimize=True).astype(np.float32)

                del E, x
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # normalizzazione non-bias e ridge per stabilità numerica
        cov /= float(max(1, N - 1))
        cov += (RIDGE * np.eye(d, dtype=np.float32))[:, :, None]  # aggiungi RIDGE a ogni Σ_l

        # ---------- SALVA PICKLE (solo training) ----------
        train_payload = {
            "version": 1,
            "cfg": {
                "backbone": "wide_resnet50_2",
                "padim_d": int(d),
                "ridge": float(RIDGE),
                "img_size": int(IMG_SIZE),
                "seed": 1024,
                "train_tag": TRAIN_TAG,
            },
            "mean": mean,                                          # (d, L)
            "cov":  cov,                                           # (d, d, L)
            "sel_idx": np.array(sel_idx, dtype=np.int64),          # (d,)
            "shape": (int(H), int(W)),                             # (H, W)
        }
        save_split_pickle(train_payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print(">> Train feature bank salvato su pickle (mean+cov).")

    # --- variabili comode per la validazione ---
    mean    = train_payload["mean"]
    cov     = train_payload["cov"]
    sel_idx = train_payload["sel_idx"]
    H, W    = train_payload["shape"]
    L       = H * W

    print(">> Train feature bank ready (from cache or freshly computed).")

    # ======== VALIDATION ========
    raw_score_maps = []   # heatmap RAW (prima di normalizzazione)
    img_scores_list = []  # image scores (dopo normalizzazione globale)
    gt_list, gt_mask_list = [], []

    with torch.inference_mode():
        for (x, y, m) in tqdm(val_loader, desc="| feature extraction | validation |"):
            gt_list.extend(y.cpu().numpy())
            gt_mask_list.extend(m.cpu().numpy())

            _ = model(x.to(device, non_blocking=True))
            l1, l2, l3 = [t.cpu().detach() for t in outputs[:3]]
            outputs.clear()

            # concat multi-scala (leggera)
            emb_t = embedding_concat_nn(l1, l2)
            emb_t = embedding_concat_nn(emb_t, l3)                # (B, Ctot, H, W)

            # stessa selezione canali del train
            idx = torch.tensor(train_payload["sel_idx"], dtype=torch.long)
            Ht, Wt = train_payload["shape"]
            emb_t = torch.index_select(emb_t, 1, idx)             # (B, d, H, W)
            Bv, dv, Hc, Wc = emb_t.shape
            assert (Ht, Wt) == (Hc, Wc)

            # -> NumPy per Mahalanobis (per-pixel) con Cholesky batched + tiling
            emb_np_v = emb_t.view(Bv, dv, Hc * Wc).numpy().astype(np.float32)     # (B, d, L)
            mean_v   = train_payload["mean"]                                      # (d, L)
            cov_v    = train_payload["cov"]                                       # (d, d, L)

            Lloc = Hc * Wc
            TILE = 256  # prova 512 se hai RAM
            dist2_LB = np.empty((Lloc, Bv), dtype=np.float32)

            for l0 in range(0, Lloc, TILE):
                l1_ = min(l0 + TILE, Lloc)
                t = l1_ - l0

                # diffs: (t, Bv, d)
                diffs_t = np.transpose(emb_np_v[:, :, l0:l1_], (2, 0, 1)) - mean_v[:, l0:l1_].T[:, None, :]

                # covariances: (t, d, d) + piccola ridge extra per sicurezza
                cov_t = np.transpose(cov_v[:, :, l0:l1_], (2, 0, 1)).copy()
                eps = 1e-2
                cov_t += eps * np.eye(dv, dtype=np.float32)[None, :, :]

                # torch batched cholesky solve
                cov_t_t = torch.from_numpy(cov_t)            # (t,d,d)
                diffs_t_t = torch.from_numpy(diffs_t)        # (t,Bv,d)

                Lfac = torch.linalg.cholesky(cov_t_t)        # (t,d,d)
                diffsT = diffs_t_t.transpose(1, 2).contiguous()  # (t,d,Bv)
                sol = torch.cholesky_solve(diffsT, Lfac)         # (t,d,Bv)

                dist2_tB = (diffsT * sol).sum(dim=1)             # (t,Bv)
                dist2_LB[l0:l1_, :] = dist2_tB.cpu().numpy().astype(np.float32)

            # back to (Bv, Hc, Wc)
            dist_arr = np.sqrt(dist2_LB.T).astype(np.float32).reshape(Bv, Hc, Wc)

            # upsample + gaussian (solo per visual)
            dist_t  = torch.from_numpy(dist_arr).unsqueeze(1)  # (B,1,H,W)
            score_b = F.interpolate(dist_t, size=IMG_SIZE, mode='bilinear',
                                    align_corners=False).squeeze(1).numpy()
            for i in range(score_b.shape[0]):
                score_b[i] = gaussian_filter(score_b[i], sigma=GAUSSIAN_SIGMA)

            # accumulo heatmap RAW
            raw_score_maps.extend([score_b[i] for i in range(score_b.shape[0])])

            # cleanup
            del l1, l2, l3, emb_t, emb_np_v, dist_t, dist_arr, score_b, x
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- normalizzazione SOLO per image score ----
    raw_score_maps = np.asarray(raw_score_maps, dtype=np.float32)   # (N, H, W)
    smax, smin = raw_score_maps.max(), raw_score_maps.min()
    scores_norm = (raw_score_maps - smin) / (smax - smin + 1e-12)   # (N, H, W) solo per image-score

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

    # Visualizzazione griglia su validation (opzionale)
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
        vis=True,
        vis_ds_or_loader=val_loader.dataset
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")


if __name__ == "__main__":
    main()
