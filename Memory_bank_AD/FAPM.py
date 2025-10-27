# FAPM_adaptive.py — FAPM fedele al paper: memory per-layer, per-patch, coreset adattivo + matching co-locato
import math
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.ndimage import gaussian_filter

from data_loader import build_ad_datasets, make_loaders, save_split_pickle, load_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report
from view_utils import show_dataset_images, show_validation_grid_from_loader

# ----------------- CONFIG -----------------
METHOD = "FAPM_ADAPT"
CODICE_PEZZO = "PZ5"

# Split come negli altri metodi
TRAIN_POSITIONS = ["pos1"]
VAL_GOOD_PER_POS = 0
VAL_GOOD_SCOPE  = ["pos1"]
VAL_FAULT_SCOPE = ["pos1"]
GOOD_FRACTION   = 1.0

# backbone & features
IMG_SIZE  = 224
SEED      = 42
GAUSSIAN_SIGMA = 3

# FAPM — patch memory per layer (paper-like)
LAYER_NAMES = ("layer1","layer2","layer3")   # tre livelli della piramide
L2NORMALIZE = True                           # l2 sui canali (consigliato)

# Coreset adattivo per patch
MEM_AVG_PER_LOC = 4        # #centroidi medi per cella (per layer)
MEM_MIN_PER_LOC = 1        # minimo per cella
MEM_MAX_PER_LOC = 8        # massimo per cella
FF_BLOCK = 2048            # blocco per cdist nel farthest-first

# Inference / matching
K_PATCH   = 3              # top-K dei centroidi per patch (media dei k min)
BLOCK_Q   = 4096           # blocco query (patch) per accelerare cdist verso memoria
DEVICE_CDIST = "cuda" if torch.cuda.is_available() else "cpu"

# Image-level
IMG_SCORE_POOL = "max"     # "max" | "p99" | "mean"

# visual
VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True
BATCH_SIZE = 32
# ------------------------------------------


# ---------- utils ----------
def set_all_seeds(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def l2norm(x, dim=1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def pool_image_score(arr, mode="max"):
    if mode == "max": return float(np.max(arr))
    if mode == "p99": return float(np.percentile(arr, 99))
    return float(np.mean(arr))

@torch.no_grad()
def extract_feats(model, loader, device):
    """Estrae e ritorna tensor per ciascun layer di LAYER_NAMES + avgpool (solo per debug)."""
    outputs = []
    def hook(_m,_i,o): outputs.append(o)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    out = OrderedDict([(n, []) for n in (*LAYER_NAMES, "avgpool")])
    with torch.inference_mode():
        for x, _, _ in tqdm(loader, desc='| feature extraction |'):
            _ = model(x.to(device, non_blocking=True))
            for k, v in zip(out.keys(), outputs):
                out[k].append(v.detach().cpu())
            outputs.clear()
    for k in out: out[k] = torch.cat(out[k], dim=0)  # su CPU
    return out

@torch.no_grad()
def farthest_first_per_cell(X: torch.Tensor, m: int) -> torch.Tensor:
    """
    Farthest-first su un set X (N,C) CPU float32. Ritorna indici (m,).
    Se N <= m -> restituisce tutti.
    """
    N = X.size(0)
    if m >= N or N == 0:
        return torch.arange(N, dtype=torch.long)
    # inizializza con un punto random
    sel = [np.random.randint(0, N)]
    # distanza minima dal set selezionato
    dmin = torch.cdist(X[sel], X, p=2.0)[0]  # (N,)
    # iterazioni
    for _ in range(1, m):
        i = int(torch.argmax(dmin).item())
        sel.append(i)
        # aggiorna dmin
        dmin = torch.minimum(dmin, torch.cdist(X[i:i+1], X, p=2.0)[0])
    return torch.tensor(sel, dtype=torch.long)

def allocate_adaptive_counts(disp_map: torch.Tensor, avg=MEM_AVG_PER_LOC,
                             mmin=MEM_MIN_PER_LOC, mmax=MEM_MAX_PER_LOC):
    """
    disp_map: (H,W) dispersion (>=0). Restituisce m_hw (H,W) interi adattivi.
    Strategia: m_hw ~ avg * (disp / mean_disp), clamp [mmin,mmax].
    """
    H, W = disp_map.shape
    d = disp_map.float()
    mean_d = torch.clamp(d.mean(), min=1e-12)
    m = (d / mean_d) * float(avg)
    m = torch.clamp(torch.round(m), min=mmin, max=mmax).to(torch.long)  # (H,W)
    return m

def build_layer_memory_bank(feats_l: torch.Tensor, l2norm_feats=True) -> dict:
    """
    Costruisce la memory per UN layer:
      input: feats_l (N, C, H, W) CPU
      output: dict con:
        - mem:     (M_total, C) tutti i centroidi concatenati
        - offsets: (H, W, 2) start/end indice in mem per ciascuna cella
        - C, H, W, counts_per_loc (H,W)
    """
    if l2norm_feats:
        feats_l = l2norm(feats_l, dim=1)

    N, C, H, W = feats_l.shape
    feats_l = feats_l.permute(0,2,3,1).contiguous().view(N, H*W, C)  # (N, HW, C)

    # --- dispersion per cella (std canale-wise medio) ---
    # std sui N campioni, poi media sui canali -> dispersione scalare
    std_hw_c = torch.std(feats_l, dim=0, unbiased=False)        # (HW, C)
    disp_hw  = std_hw_c.mean(dim=1).view(H, W)                  # (H,W)

    # --- allocazione #centroidi per cella ---
    m_hw = allocate_adaptive_counts(disp_hw)                    # (H,W) int

    # --- per cella: farthest-first su N vettori (N<=~pochi), raccogli centroidi ---
    mem_chunks = []
    offsets = torch.zeros((H, W, 2), dtype=torch.int64)
    cursor = 0

    for h in range(H):
        for w in range(W):
            X = feats_l[:, h*W + w, :]     # (N, C)
            m = int(m_hw[h, w].item())
            # farthest-first sui N vettori (CPU)
            idx = farthest_first_per_cell(X.float(), m)  # (m',) m'<=m se N<m
            Ym = X[idx]                                  # (m', C)
            mem_chunks.append(Ym)
            start = cursor
            cursor += Ym.shape[0]
            offsets[h, w, 0] = start
            offsets[h, w, 1] = cursor

    mem = torch.cat(mem_chunks, dim=0).contiguous() if len(mem_chunks) else torch.empty((0, C))
    payload = {
        "mem": mem,                   # (M_total, C) CPU
        "offsets": offsets,           # (H, W, 2) start/end
        "counts_per_loc": m_hw,       # (H, W) #centroidi allocati
        "C": int(C), "H": int(H), "W": int(W),
        "l2norm": bool(l2norm_feats),
    }
    return payload

def build_fapm_adaptive_memory(train_feats: OrderedDict, l2norm_feats=L2NORMALIZE) -> dict:
    """
    Costruisce la memory bank FAPM per TUTTI i layer in LAYER_NAMES.
    Ritorna dict { layer_name: layer_payload, ... }
    """
    mb = {}
    for lname in LAYER_NAMES:
        print(f"[FAPM] build memory for {lname} ...")
        mb[lname] = build_layer_memory_bank(train_feats[lname].contiguous(), l2norm_feats)
        print(f"  -> mem size = {mb[lname]['mem'].shape[0]} (C={mb[lname]['C']}, HxW={mb[lname]['H']}x{mb[lname]['W']})")
    return mb

@torch.no_grad()
def infer_score_map_one_image(test_feats_one: OrderedDict, mbank: dict,
                              k_patch=K_PATCH, img_size=IMG_SIZE, gaussian_sigma=GAUSSIAN_SIGMA):
    """
    test_feats_one: dict {layer1:(1,C,H,W), layer2:..., layer3:...} per UNA immagine (CPU)
    mbank: {layer_name: {'mem','offsets','C','H','W',...}, ...}
    Ritorna: heatmap fused (img_size,img_size) numpy
    """
    layer_maps = []

    for lname in LAYER_NAMES:
        tf = test_feats_one[lname]             # (1, C, H, W) CPU
        if mbank[lname]["l2norm"]:
            tf = l2norm(tf, dim=1)
        _, C, H, W = tf.shape
        assert (H, W) == (mbank[lname]["H"], mbank[lname]["W"]), "Mismatch (H,W) tra test e memoria"

        q = tf.view(C, H*W).transpose(0,1).contiguous()   # (HW, C)
        mem = mbank[lname]["mem"]                         # (M_total, C)
        offs = mbank[lname]["offsets"]                    # (H, W, 2)

        # score per locazione
        s_loc = torch.empty((H*W,), dtype=torch.float32)

        # calcolo co-locato: per ogni (h,w) prendo solo i centroidi di quella locazione
        # (loop per evitare cdist con maschere variabili; con pochi centroidi/cella è veloce)
        device = torch.device(DEVICE_CDIST)
        mem_dev = mem.to(device, non_blocking=True)

        for hw in range(H*W):
            h = hw // W
            w = hw % W
            a, b = int(offs[h, w, 0].item()), int(offs[h, w, 1].item())
            if a == b:
                # nessun centroide (può capitare se N=0, quasi impossibile) -> score 0
                s_loc[hw] = 0.0
                continue
            qi = q[hw:hw+1].to(device)          # (1,C)
            Mi = mem_dev[a:b]                   # (m,C)
            d = torch.cdist(qi, Mi, p=2.0)      # (1,m)
            if k_patch >= 2 and Mi.size(0) >= k_patch:
                v, _ = torch.topk(d, k=k_patch, dim=1, largest=False)
                s = v.mean(dim=1)[0]
            else:
                s = d.min(dim=1).values[0]
            s_loc[hw] = s.float().cpu()

        # rimappa a (H,W) e upsample
        m = s_loc.view(1,1,H,W)
        m = F.interpolate(m, size=img_size, mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        if gaussian_sigma and gaussian_sigma > 0:
            m = gaussian_filter(m, sigma=gaussian_sigma)
        layer_maps.append(m)

    # fusione layer-wise (media)
    fused = np.mean(np.stack(layer_maps, axis=0), axis=0).astype(np.float32)
    return fused


# ============================== MAIN ==============================
def main():
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

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

    # ====== FEATURE CACHE (train/val) ======
    try:
        feats_train = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        print("[cache] Train features loaded.")
    except FileNotFoundError:
        print("[cache] Extracting train features...")
        feats_train = extract_feats(model, train_loader, device)
        save_split_pickle(feats_train, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    try:
        val_pack = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)
        feats_val, gt_list = val_pack["features"], val_pack["labels"]
        print("[cache] Validation features loaded.")
    except FileNotFoundError:
        print("[cache] Extracting validation features...")
        feats_val = extract_feats(model, val_loader, device)
        gt_list = []
        for _, y, _ in val_loader: gt_list.extend(y.cpu().numpy())
        save_split_pickle({"features": feats_val, "labels": np.array(gt_list, dtype=np.int64)},
                          CODICE_PEZZO, TRAIN_TAG, split="validation", method=METHOD)

    # ====== MEMORY BANK per layer (cache) ======
    MB_METHOD = f"{METHOD}_MBANK_ADAPT"
    try:
        mbank = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=MB_METHOD)
        print("[cache] FAPM adaptive memory loaded.")
    except FileNotFoundError:
        print("[cache] Building FAPM adaptive memory (per-layer, per-patch)...")
        mbank = build_fapm_adaptive_memory(feats_train, l2norm_feats=L2NORMALIZE)
        save_split_pickle(mbank, CODICE_PEZZO, TRAIN_TAG, split="train", method=MB_METHOD)

    # ====== INFERENZA HEATMAPS ======
    score_map_list = []
    N = feats_val[LAYER_NAMES[0]].shape[0]
    for i in tqdm(range(N), desc="| FAPM (adaptive, co-located) inference |"):
        one = OrderedDict(
            [(lname, feats_val[lname][i:i+1]) for lname in LAYER_NAMES]
        )
        s = infer_score_map_one_image(one, mbank, k_patch=K_PATCH, img_size=IMG_SIZE, gaussian_sigma=GAUSSIAN_SIGMA)
        score_map_list.append(s)

    # ====== IMAGE-LEVEL SCORE ======
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

    # Visualizzazione griglia (opzionale)
    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds_img,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True, overlay=True, overlay_alpha=0.45
        )

    # ====== Valutazione pixel-level ======
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
