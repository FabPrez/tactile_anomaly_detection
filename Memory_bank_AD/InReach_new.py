# InReaCh_official_full_adapted.py — Implementazione fedele (paper+repo) adattata al tuo dataset
# Dipendenze: torch, torchvision, faiss-cpu (o faiss-gpu), numpy, scipy, scikit-learn, tqdm
# Integra le tue utility: data_loader (build_ad_datasets, make_loaders), ad_analysis, view_utils
# Novità vs. versione "min":
#  - Algoritmo 1 (symmetrical optimal pairing) con bookkeeping globale e scelta minima per (img_t, patch_t)
#  - Trimming in 2 stadi: (i) outlier per sigma sul raggio, (ii) channel span minimo (copertura immagini distinte)
#  - Positional Embedding condizionale via transpose-test (quick A/B sul primo seed): abilita PE se migliora la distanza media
#  - Aggregatore locale A_v (eq. (1)) con finestra (h,w), implementato con avg_pool2d su feature map a bassa profondità
#  - Protocollo immagini 256→224 (resize + center-crop) per comparabilità (se il tuo loader non lo fa già)

import os, math, random, time
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.ndimage import gaussian_filter

# --- tue utility ---
from data_loader import build_ad_datasets, make_loaders
from view_utils import show_dataset_images, show_validation_grid_from_loader
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# --- FAISS ---
try:
    import faiss
except Exception as e:
    raise ImportError("InReaCh richiede FAISS. Installa: pip install faiss-cpu (oppure conda: faiss-cpu/faiss-gpu).") from e

# forza default su float32 per sicurezza
torch.set_default_dtype(torch.float32)

# ---------------- CONFIG ----------------
METHOD = "INREACH_OFFICIAL"
CODICE_PEZZO = "PZ3"

TRAIN_POSITIONS = ["pos2"]
VAL_GOOD_PER_POS = 0
VAL_GOOD_SCOPE   = ["pos2"]
VAL_FAULT_SCOPE  = ["pos2"]
GOOD_FRACTION    = 1.0

IMG_SIZE  = 224
SEED      = 42

# InReaCh (paper/repo)
ASSOC_DEPTH        = 10      # D
MIN_CHANNEL_LENGTH = 3       # lunghezza minima dopo trimming-1
MIN_CHANNEL_SPAN   = 3       # numero minimo di immagini distinte nel canale (trimming-2)
MAX_CHANNEL_STD    = 5.0     # coeff. per soglia raggio vs sigma
FILTER_SIZE        = 13

# Aggregatore locale (eq. 1)
AGG_WINDOW_H = 3
AGG_WINDOW_W = 3
AGG_STRIDE   = 1
PRE_EMBED_DIM = 1024
TARGET_EMBED_DIM = 1024

# Positional Embedding
PE_MAX_W = 0.25      # peso massimo
PE_TEST_SAMPLES = 64 # numero di patch per test rapido

# Visual
VIS_VALID_DATASET               = False
VIS_PREDICTION_ON_VALID_DATASET = True

# ---------------- Utils ----------------
def super_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def cdist_L2_colwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Distanza euclidea tra colonne: a:(F,La), b:(F,Lb) → (La,Lb)
    """
    if a.dtype != torch.float32: a = a.to(torch.float32)
    if b.dtype != torch.float32: b = b.to(torch.float32)
    return torch.cdist(a.T.contiguous(), b.T.contiguous())

# ---------------- Backbone & Feature taps ----------------
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


def load_wide_resnet_50(return_nodes: Dict[str, str] = None):
    try:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2',
                               Wide_ResNet50_2_Weights.IMAGENET1K_V1,
                               force_reload=False, verbose=False)
    except Exception:
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    if return_nodes is not None:
        model = create_feature_extractor(model, return_nodes=return_nodes)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

# --- trasformazioni 256→224 se servono ---

def _resize_center_crop_uint8(img: np.ndarray, side_crop: int = 224) -> np.ndarray:
    # img: HxWxC uint8
    H, W, C = img.shape
    # resize lato minore a 256 mantenendo AR
    if min(H, W) != 256:
        scale = 256.0 / min(H, W)
        newH, newW = int(round(H * scale)), int(round(W * scale))
        # bilinear con torch per coerenza
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
        t = F.interpolate(t, size=(newH, newW), mode='bilinear', align_corners=False)
        img = (t.squeeze(0).permute(1,2,0).clamp(0,1) * 255.0).byte().cpu().numpy()
        H, W, C = img.shape
    # center crop 224x224
    y0 = (H - side_crop) // 2
    x0 = (W - side_crop) // 2
    return img[y0:y0+side_crop, x0:x0+side_crop, :]


# ---------------- Feature Descriptor con A_v e PE condizionale ----------------
class FeatureDescriptor(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 taps: List[str],
                 agg_win: Tuple[int,int] = (3,3),
                 agg_stride: int = 1,
                 pre_dim: int = 1024,
                 target_dim: int = 1024,
                 pe_weight: float = 0.0):
        super().__init__()
        self.model = model
        self.taps = taps
        self.agg_win = agg_win
        self.agg_stride = agg_stride
        self.pre_dim = pre_dim
        self.target_dim = target_dim
        self.pe_weight = pe_weight

        # probe per dimensioni
        device = next(self.model.parameters()).device
        dummy = torch.zeros(1,3,224,224, device=device, dtype=torch.float32)
        with torch.no_grad():
            feats = self.model(dummy)
        self.feat_dims = {k: v.shape[1] for k, v in feats.items() if k in taps}

        # moduli	note: pre-aggregazione: riduzione a pre_dim per livello
        self.pre_mappers = torch.nn.ModuleDict({
            k: torch.nn.Sequential(
                torch.nn.Conv2d(self.feat_dims[k], self.feat_dims[k], kernel_size=1, bias=False),
                torch.nn.AdaptiveAvgPool2d(1), # solo per normalizzare scala canali (non spatial)
            ) for k in taps
        })
        # proiezione 1D per livello tramite adaptive_avg_pool1d
        self.proj_per_level = torch.nn.ModuleDict({
            k: torch.nn.Sequential(
                torch.nn.Conv1d(1, 1, kernel_size=1, bias=False)
            ) for k in taps
        })
        # proiezione finale a target_dim (concat livelli → avg1d)
        self.final_proj = torch.nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.to(device)

    def _imagenet_norm(self, img_uint8: np.ndarray, device):
        x = img_uint8.astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)

    def _add_positional_embedding(self, DHW: torch.Tensor) -> torch.Tensor:
        # DHW: (B,D,H,W) → aggiunge (x,y) normalizzati con peso pe_weight
        if self.pe_weight is None or self.pe_weight <= 0.0:
            return DHW
        B, D, H, W = DHW.shape
        device = DHW.device
        ys = torch.linspace(-1, 1, steps=H, device=device, dtype=torch.float32).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1, 1, steps=W, device=device, dtype=torch.float32).view(1, 1, 1, W).expand(B, 1, H, W)
        DHW = torch.cat([DHW, self.pe_weight * xs, self.pe_weight * ys], dim=1)
        return DHW

    @torch.no_grad()
    def _forward_feats(self, img_uint8: np.ndarray) -> Dict[str, torch.Tensor]:
        # 256→224 protocollo
        img_uint8 = _resize_center_crop_uint8(img_uint8, 224)
        device = next(self.model.parameters()).device
        x = self._imagenet_norm(img_uint8, device)
        feats = self.model(x)
        # seleziona taps
        feats = {k: v for k, v in feats.items() if k in self.taps}
        return feats

    @torch.no_grad()
    def _aggregate_levels(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Local aggregator A_v: avg_pool2d finestra (h,w), poi unfold con stride=agg_stride, poi riduzione per livello
        B = 1
        pooled_levels = []
        for k in self.taps:
            f = feats[k]  # (1,C,H,W)
            f = F.avg_pool2d(f, kernel_size=self.agg_win, stride=1, padding=(self.agg_win[0]//2, self.agg_win[1]//2))
            # patchify con stride agg_stride
            unfold = torch.nn.Unfold(kernel_size=1, stride=self.agg_stride, padding=0)
            u = unfold(f)  # (B, C*1*1, L) = (1, C, L)
            u = u.view(B, f.shape[1], -1)  # (1, C, L)
            # (C,L) → (L, C)
            u = u.squeeze(0).permute(1,0).contiguous()
            # riduzione a pre_dim per livello: adaptive_avg_pool1d su (L, C)
            # risagoma per usare Conv1d
            u1 = u.view(u.shape[0], 1, -1)  # (L,1,C)
            # proiezione (identità con conv1x1) + adaptive avg a pre_dim
            u1 = F.adaptive_avg_pool1d(u1, self.pre_dim)
            # (L,pre_dim)
            u1 = u1.squeeze(1)
            pooled_levels.append(u1)
        # stack livelli: (L, n_levels, pre_dim)
        Lref = pooled_levels[0].shape[0]
        stack = torch.stack(pooled_levels, dim=1)  # (L, n_levels, pre_dim)
        # concat livelli e proiezione a target_dim
        concat = stack.reshape(Lref, 1, -1)       # (L,1,n_levels*pre_dim)
        concat = F.adaptive_avg_pool1d(concat, self.target_dim).squeeze(1)  # (L, D)
        # rimappa a (1,D,H,W)
        side = int(math.sqrt(Lref))
        DHW = concat.T.view(1, self.target_dim, side, side)
        DHW = self._add_positional_embedding(DHW)
        # flatten a (D', L)
        DHW = DHW.view(1, DHW.shape[1], -1)  # (1,D',L)
        return DHW

    @torch.no_grad()
    def generate_descriptors(self, images: List[np.ndarray]) -> torch.Tensor:
        outs = []
        for img in tqdm(images, ncols=100, desc='Gen Feature Descriptors'):
            feats = self._forward_feats(img)
            DHW = self._aggregate_levels(feats)  # (1,D',L)
            outs.append(DHW.cpu())
        return torch.cat(outs, dim=0).to(torch.float32)  # (N,D',L)


# ---------------- InReaCh core (Algoritmo 1 + trimming + NN) ----------------
class InReaCh:
    def __init__(self,
                 images: List[np.ndarray],
                 model: torch.nn.Module,
                 return_nodes: Dict[str,str],
                 assoc_depth: int = 10,
                 min_channel_length: int = 3,
                 min_channel_span: int = 3,
                 max_channel_std: float = 5.0,
                 filter_size: int = 13):
        self.images = images
        self.assoc_depth = assoc_depth
        self.min_channel_length = min_channel_length
        self.min_channel_span = min_channel_span
        self.max_channel_std = max_channel_std
        self.filter_size = filter_size

        # Transpose-test rapido per decidere PE
        self.pe_weight = self._transpose_test(model, return_nodes)

        # Generatore descrittori con PE deciso
        self.fd = FeatureDescriptor(
            model=model,
            taps=list(return_nodes.values()),
            agg_win=(AGG_WINDOW_H, AGG_WINDOW_W),
            agg_stride=AGG_STRIDE,
            pre_dim=PRE_EMBED_DIM,
            target_dim=TARGET_EMBED_DIM,
            pe_weight=self.pe_weight,
        )
        self.patches = self.fd.generate_descriptors(self.images)  # (N,D,L)
        self.cpu_patches = self.patches.cpu().numpy().astype(np.float32, copy=False)

        self._build_channels()

    @torch.no_grad()
    def _transpose_test(self, model, return_nodes) -> float:
        # Prova senza PE vs con PE su una coppia (seed0 vs img1), confronta distanza media MNN su PE on/off
        if len(self.images) < 2:
            return 0.0
        device = next(model.parameters()).device
        fd_off = FeatureDescriptor(model, list(return_nodes.values()), (AGG_WINDOW_H,AGG_WINDOW_W), AGG_STRIDE,
                                   PRE_EMBED_DIM, TARGET_EMBED_DIM, pe_weight=0.0)
        fd_on  = FeatureDescriptor(model, list(return_nodes.values()), (AGG_WINDOW_H,AGG_WINDOW_W), AGG_STRIDE,
                                   PRE_EMBED_DIM, TARGET_EMBED_DIM, pe_weight=PE_MAX_W)
        d0 = fd_off.generate_descriptors(self.images[:2])  # (2,D,L)
        d1 = fd_on.generate_descriptors(self.images[:2])
        # col-wise MNN distance media su subset di patch
        def mean_mnn(x):
            a, b = x[0], x[1]  # (D,L)
            L = a.shape[1]
            sel = torch.linspace(0, L-1, steps=min(PE_TEST_SAMPLES, L)).long()
            da = cdist_L2_colwise(a[:, sel], b)
            db = cdist_L2_colwise(b[:, sel], a)
            # mutual
            idx_a = torch.argmin(da, dim=1)
            idx_b = torch.argmin(db, dim=1)
            ok = (torch.arange(sel.numel()) == torch.gather(idx_b, 1, idx_a.view(-1,1)).squeeze(1))
            vals = torch.gather(da, 1, idx_a.view(-1,1)).squeeze(1)
            if ok.any():
                vals = vals[ok]
            return float(vals.mean().cpu().item())
        m_off = mean_mnn(d0)
        m_on  = mean_mnn(d1)
        return PE_MAX_W if m_on < m_off else 0.0

    @torch.no_grad()
    def _build_channels(self):
        N, D, L = self.patches.shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        order = list(range(N))
        random.shuffle(order)
        seeds = order[:min(self.assoc_depth, N)]

        # bookkeeping globale: per ogni (img_t, patch_t) teniamo il best (dist, seed, src_img, src_patch)
        best = {
            (ti, pt): [np.inf, -1, -1, -1]  # dist, seed, src_img, src_patch
            for ti in range(N) for pt in range(L)
        }

        # Algoritmo 1: per ogni seed, confronta con tutte le altre immagini e applica symmetrical optimal pairing
        for s in tqdm(seeds, desc='Build channels [Algorithm 1]', ncols=100):
            tgt = self.patches[s].to(device)  # (D,L)
            for j in range(N):
                if j == s: continue
                src = self.patches[j].to(device)  # (D,L)
                # distanza per blocchi per non esplodere memoria
                block = 2048
                for t0 in range(0, L, block):
                    t1 = min(t0+block, L)
                    for s0 in range(0, L, block):
                        s1 = min(s0+block, L)
                        d = cdist_L2_colwise(tgt[:, t0:t1], src[:, s0:s1])  # (Lt,Ls)
                        # argmin per colonna/righe
                        mins_t, argmin_t = torch.min(d, dim=1)  # per ogni patch tgt→ best src
                        mins_s, argmin_s = torch.min(d, dim=0)  # per ogni patch src→ best tgt
                        # mutual: t ↔ s
                        for t_idx_local in range(mins_t.shape[0]):
                            s_idx_local = int(argmin_t[t_idx_local].item())
                            # verifica simmetria: il best di s_idx_local deve essere t_idx_local
                            if int(argmin_s[s_idx_local].item()) == t_idx_local:
                                dist_val = float(mins_t[t_idx_local].item())
                                t_idx = t0 + t_idx_local
                                s_idx = s0 + s_idx_local
                                key = (s, t_idx)  # canale indicizzato dal patch del seed s
                                # aggiornamento globale del best per (seed s, patch t_idx)
                                cur = best[(s, t_idx)]
                                if dist_val < cur[0]:
                                    best[(s, t_idx)] = [dist_val, s, j, s_idx]

        # Ricostruisci i canali a partire da best
        channels: Dict[Tuple[int,int], List[Tuple[np.ndarray,int,int]]] = {}
        for (ti, pt), (dist, seed, src_img, src_pt) in best.items():
            if seed < 0 or src_img < 0:
                continue
            cname = (seed, pt)
            if cname not in channels:
                # il primo elemento del canale è il centro (seed)
                center_patch = self.cpu_patches[seed, :, pt]
                channels[cname] = [(center_patch, seed, pt)]
            # append associato
            channels[cname].append((self.cpu_patches[src_img, :, src_pt], src_img, src_pt))

        # Trimming 1: outlier su raggio (sigma)
        trimmed_channels: Dict[Tuple[int,int], List[Tuple[np.ndarray,int,int]]] = {}
        for cname, plist in tqdm(channels.items(), desc='Trimming-1 (sigma)', ncols=100):
            if len(plist) <= 1:
                continue
            patches = np.stack([p[0] for p in plist], axis=0)  # (M,D)
            center = patches[0]
            radii = np.sqrt(((patches - center)**2).sum(axis=1))
            sigma = np.std(radii)
            keep = [plist[i] for i in range(len(plist)) if radii[i] <= (self.max_channel_std * sigma + 1e-12)]
            if len(keep) >= self.min_channel_length:
                trimmed_channels[cname] = keep

        # Trimming 2: channel span minimo (immagini distinte coperte)
        nominal_points = []
        for cname, plist in tqdm(trimmed_channels.items(), desc='Trimming-2 (span)', ncols=100):
            imgs = set([img for (_, img, _) in plist])
            if len(imgs) >= self.min_channel_span:
                nominal_points.extend([p[0] for p in plist])

        if len(nominal_points) == 0:
            raise RuntimeError("Dopo il trimming non restano nominal points: aumenta D, riduci MAX_CHANNEL_STD o Span.")

        base_np = np.array(nominal_points, dtype=np.float32, order='C')
        d = int(base_np.shape[1])
        if torch.cuda.is_available():
            try:
                self.nn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d, faiss.GpuIndexFlatConfig())
            except Exception:
                self.nn = faiss.IndexFlatL2(d)
        else:
            self.nn = faiss.IndexFlatL2(d)
        self.nn.add(base_np.astype(np.float32, copy=False))

    @torch.no_grad()
    def predict(self, test_images: List[np.ndarray]) -> List[np.ndarray]:
        t_patches = self.fd.generate_descriptors(test_images)  # (N,D',L)
        outs = []
        for i in tqdm(range(t_patches.shape[0]), desc='Predict', ncols=100):
            q = t_patches[i].permute(1,0).contiguous().cpu().numpy().astype(np.float32)  # (L,D')
            dist, _ = self.nn.search(q, 1)
            side = int(np.sqrt(dist.shape[0]))
            dist2d = dist[:,0].reshape(side, side).astype(np.float32)
            scale = test_images[i].shape[0] // side
            dist2d = dist2d.repeat(scale, axis=0).repeat(scale, axis=1)
            outs.append(gaussian_filter(dist2d, FILTER_SIZE))
        return outs


# ---------------- Helper tensori→uint8 ----------------
def _tensor_batch_to_uint8_images(x: torch.Tensor) -> List[np.ndarray]:
    x = x.detach().cpu()
    if x.dtype != torch.uint8:
        x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
    return [im for im in x.permute(0, 2, 3, 1).numpy()]


def _tensor_batch_to_uint8_masks(m: torch.Tensor) -> List[np.ndarray]:
    m = m.detach().cpu()
    if m.ndim == 3:
        m = m.unsqueeze(-1)
    if m.dtype != torch.uint8:
        m = (m > 0).to(torch.uint8) * 255
    return [mk for mk in m.numpy()]


# ---------------- MAIN ----------------
def main():
    super_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device:", device)

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

    # immagini GOOD train
    train_imgs: List[np.ndarray] = []
    with torch.inference_mode():
        for xb, yb, mb in tqdm(train_loader, desc="| collect train imgs |"):
            sel = (yb == 0)
            if sel.any():
                train_imgs.extend(_tensor_batch_to_uint8_images(xb[sel]))

    if len(train_imgs) == 0:
        raise RuntimeError("Nessuna immagine GOOD per il train. Controlla TRAIN_POSITIONS/GOOD_FRACTION.")

    # backbone e nodi (paper: taps su blocchi a bassa/media profondità)
    return_nodes = {
        'layer1.0.relu_2': 'L1',
        'layer1.1.relu_2': 'L2',
        'layer1.2.relu_2': 'L3',
        'layer2.0.relu_2': 'L4',
        'layer2.1.relu_2': 'L5',
        'layer2.2.relu_2': 'L6',
        'layer2.3.relu_2': 'L7',
        'layer3.1.relu_2': 'L8',
        'layer3.2.relu_2': 'L9',
        'layer3.3.relu_2': 'L10',
        'layer3.4.relu_2': 'L11',
        'layer3.5.relu_2': 'L12',
        'layer4.0.relu_2': 'L13',
    }
    model = load_wide_resnet_50(return_nodes=return_nodes)

    # InReaCh
    inreach = InReaCh(
        images=train_imgs,
        model=model,
        return_nodes=return_nodes,
        assoc_depth=ASSOC_DEPTH,
        min_channel_length=MIN_CHANNEL_LENGTH,
        min_channel_span=MIN_CHANNEL_SPAN,
        max_channel_std=MAX_CHANNEL_STD,
        filter_size=FILTER_SIZE,
    )

    # val → liste
    val_imgs: List[np.ndarray] = []
    val_masks: List[np.ndarray] = []
    val_labels: List[int] = []
    with torch.inference_mode():
        for xb, yb, mb in tqdm(val_loader, desc="| collect val imgs |"):
            val_imgs.extend(_tensor_batch_to_uint8_images(xb))
            val_masks.extend(_tensor_batch_to_uint8_masks(mb))
            val_labels.extend(yb.detach().cpu().numpy().tolist())

    # inferenza
    score_maps = inreach.predict(val_imgs)

    # image-level score
    img_scores = np.asarray([float(np.max(s)) for s in score_maps], dtype=np.float32)
    y_true    = np.asarray(val_labels, dtype=np.int32)

    fpr, tpr, thr = roc_curve(y_true, img_scores)
    auc_img = roc_auc_score(y_true, img_scores)
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/train={TRAIN_TAG}): {auc_img:.3f}")

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thr[best_idx])
    preds = (img_scores >= best_thr).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{tpr[best_idx]:.3f}  FPR:{fpr[best_idx]:.3f}")

    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True, overlay_alpha=0.45
        )

    # pixel-level (heatmap RAW)
    results = run_pixel_level_evaluation(
        score_map_list=score_maps,
        val_set=val_set,
        img_scores=img_scores.tolist(),
        use_threshold="pro",
        fpr_limit=0.01,
        vis=True,
        vis_ds_or_loader=val_loader.dataset
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")


if __name__ == "__main__":
    main()
