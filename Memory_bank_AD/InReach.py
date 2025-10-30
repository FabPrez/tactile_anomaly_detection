# InReach.py — InReaCh "repo-faithful" adattato al tuo dataset
# Dipendenze: torch, torchvision, faiss-cpu (o faiss-gpu), numpy, scipy, sklearn, tqdm
# Usa le tue utility: data_loader (build_ad_datasets, make_loaders), ad_analysis, view_utils

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
VAL_GOOD_SCOPE   = ["pos2"]     # "from_train" | "all_positions" | lista
VAL_FAULT_SCOPE  = ["pos2"]     # "train_only" | "all" | lista
GOOD_FRACTION    = 1.0

IMG_SIZE  = 224
SEED      = 42

# InReaCh (repo)
ASSOC_DEPTH        = 10
MIN_CHANNEL_LENGTH = 3
MAX_CHANNEL_STD    = 5.0              # pruning soglia
FILTER_SIZE        = 13               # smooth finale per mappa

# k-NN (repo-faithful): k=1
KNN_K = 1

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

def build_2d_sincos_pos_embed(H: int, W: int, dim: int, temperature: float = 10000.0) -> np.ndarray:
    """
    PE 2D sinusoidale: genera (H*W, dim) da concatenare alle feature.
    """
    assert dim % 2 == 0, "Usa dim pari per i PE"
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # (H,W)
    omega = np.arange(dim // 2, dtype=np.float32) / (dim // 2)
    omega = 1. / (temperature ** omega)  # (dim/2,)

    out_y = np.einsum('hw,c->hwc', grid_y.astype(np.float32), omega)  # (H,W,dim/2)
    out_x = np.einsum('hw,c->hwc', grid_x.astype(np.float32), omega)  # (H,W,dim/2)

    pe = np.concatenate([np.sin(out_y), np.cos(out_y), np.sin(out_x), np.cos(out_x)], axis=-1)  # (H,W,dim*2)
    if pe.shape[-1] >= dim:
        pe = pe[..., :dim]
    else:
        pad = dim - pe.shape[-1]
        pe = np.pad(pe, ((0,0),(0,0),(0,pad)), mode='constant')
    pe = pe.reshape(H*W, dim).astype(np.float32, copy=False)
    return pe

# ---------------- MODEL ----------------
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

def load_wide_resnet_50(return_nodes: Dict[str, str] = None):
    try:
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    except Exception:
        model = wide_resnet50_2(weights=None)
    if return_nodes is not None:
        model = create_feature_extractor(model, return_nodes=return_nodes)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

# ---------------- Feature Descriptors (repo-faithful) ----------------
class PatchMaker:
    def __init__(self, patchsize: int = 3, stride: int = 1):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, feat: torch.Tensor, return_hw=False):
        """
        feat: (B, C, H, W) -> (B, L, C, ps, ps), con padding 'same' per stride=1
        """
        ps = self.patchsize
        pad = (ps - 1) // 2
        unfolder = torch.nn.Unfold(kernel_size=ps, stride=self.stride, padding=pad, dilation=1)
        unfolded = unfolder(feat)                     # (B, C*ps*ps, L)
        B, _, L = unfolded.shape
        C = feat.size(1)
        unfolded = unfolded.reshape(B, C, ps, ps, L)  # (B, C, ps, ps, L)
        unfolded = unfolded.permute(0, 4, 1, 2, 3)    # (B, L, C, ps, ps)

        H = (feat.size(-2) + 2*pad - (ps - 1) - 1) // self.stride + 1
        W = (feat.size(-1) + 2*pad - (ps - 1) - 1) // self.stride + 1

        if return_hw:
            return unfolded, (H, W)
        return unfolded

class FeatureDescriptor(torch.nn.Module):
    """
    Pipeline:
      - estrai più livelli WRN
      - patchify ogni livello
      - riallinea ogni livello (L_i -> L_ref) con bilinear su griglia (H_ref, W_ref)
      - concatena TUTTI i livelli + PE 2D
      - proiezione con adaptive avg pool a target_dim
      - ritorna (D, L) per immagine
    """
    def __init__(self,
                 backbone: torch.nn.Module,
                 return_nodes_keys_in_order: List[str],
                 target_embed_dimension: int = 1024,
                 agg_patch_kernel: int = 3,
                 agg_patch_stride: int = 1,
                 pos_embed_dim: int = 64,  # PE 2D
                 use_positional_embeddings: bool = True):
        super().__init__()
        self.model = backbone
        self.nodes = return_nodes_keys_in_order  # <<-- lista delle OUTPUT-keys
        self.target_dim = target_embed_dimension
        self.pm = PatchMaker(patchsize=agg_patch_kernel, stride=agg_patch_stride)
        self.use_pe = use_positional_embeddings
        self.pos_dim = pos_embed_dim

    def _imagenet_norm(self, img_uint8: np.ndarray, device):
        x = img_uint8.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward_single(self, img_uint8: np.ndarray) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = self._imagenet_norm(img_uint8, device)
        feats_dict = self.model(x)  # dict {out_key: (1,C,H,W)}
        # Ordina i livelli come in return_nodes_keys_in_order (OUTPUT-keys)
        try:
            feats = [feats_dict[k].to(torch.float32) for k in self.nodes]
        except KeyError as e:
            raise KeyError(f"Chiave di output '{e.args[0]}' non presente. "
                           f"Assicurati di passare le OUTPUT-keys (list(return_nodes.values())).")

        # Patchify ogni livello e tieni HW di ciascuna griglia patch
        patch_list = []
        hw_list = []
        for f in feats:
            pf, hw = self.pm.patchify(f, return_hw=True)  # (1, L_i, C, ps, ps), (H_i,W_i)
            patch_list.append(pf)                         # (1, L_i, C, ps, ps)
            hw_list.append(hw)

        # Griglia di riferimento = quella del primo livello
        Href, Wref = hw_list[0]

        # Riallinea ogni livello alla griglia (Href, Wref) via bilinear
        aligned_levels = []
        for pf, (Hi, Wi) in zip(patch_list, hw_list):
            # (1, L_i, C, ps, ps) -> (1, C, ps, ps, Hi, Wi)
            pf6 = pf.reshape(1, Hi, Wi, pf.size(2), pf.size(3), pf.size(4)).permute(0, 3, 4, 5, 1, 2)
            base = pf6.shape
            pf2d = pf6.reshape(-1, 1, Hi, Wi)
            pf2d = F.interpolate(pf2d, size=(Href, Wref), mode='bilinear', align_corners=False)
            # back to (1, L_ref, C, ps, ps)
            pf6r = pf2d.reshape(base[0], base[1], base[2], base[3], Href, Wref).permute(0, 4, 5, 1, 2, 3)
            pf_lr = pf6r.reshape(1, Href*Wref, pf.size(2), pf.size(3), pf.size(4))  # (1, L_ref, C, ps, ps)
            aligned_levels.append(pf_lr)

        # Per livello -> (L_ref, C*ps*ps)
        levels_flat = []
        for pf in aligned_levels:
            Lref = pf.size(1)
            levels_flat.append(pf.squeeze(0).reshape(Lref, -1))  # (L_ref, Fi)

        # Concat tra livelli -> (L_ref, sum(Fi))
        feats_cat = torch.cat(levels_flat, dim=1)  # (L_ref, Ftot)

        # Aggiungi PE 2D (sin/cos) concatenata
        if self.use_pe and self.pos_dim > 0:
            pe = build_2d_sincos_pos_embed(Href, Wref, self.pos_dim)  # (L_ref, pos_dim)
            pe_t = torch.from_numpy(pe).to(feats_cat.device, dtype=torch.float32)
            feats_cat = torch.cat([feats_cat, pe_t], dim=1)  # (L_ref, Ftot+pos_dim)

        # Proiezione con adaptive avg pooling a target_dim
        proj = F.adaptive_avg_pool1d(feats_cat.unsqueeze(1), self.target_dim).squeeze(1)  # (L_ref, D)

        # Rimappa a (D, L)
        emb = proj.permute(1, 0).contiguous()  # (D, L_ref)
        return emb

    @torch.no_grad()
    def generate_descriptors(self, images: List[np.ndarray], quiet: bool = False) -> torch.Tensor:
        outs = []
        for img in tqdm(images, ncols=100, desc='Gen Feature Descriptors', disable=quiet):
            emb = self.forward_single(img)   # (D, L)
            outs.append(emb.unsqueeze(0).cpu())
        return torch.cat(outs, dim=0).to(torch.float32)  # (N, D, L)

# ---------------- InReaCh (repo-faithful) ----------------
class InReaCh:
    def __init__(self,
                 images: List[np.ndarray],
                 model: torch.nn.Module,
                 return_nodes_keys_in_order: List[str],
                 assoc_depth: int = 10,
                 min_channel_length: int = 3,
                 max_channel_std: float = 5.0,
                 filter_size: int = 13,
                 use_positional_embeddings: bool = True,
                 pos_embed_dim: int = 64):
        self.images = images
        self.model = model
        self.assoc_depth = assoc_depth
        self.min_channel_length = min_channel_length
        self.max_channel_std = max_channel_std
        self.filter_size = filter_size

        # feature descriptor repo-faithful
        self.fd = FeatureDescriptor(
            backbone=model,
            return_nodes_keys_in_order=return_nodes_keys_in_order,  # <<-- OUTPUT-keys
            target_embed_dimension=1024,
            agg_patch_kernel=3,
            agg_patch_stride=1,
            pos_embed_dim=pos_embed_dim,
            use_positional_embeddings=use_positional_embeddings
        )

        # (N, D, L) train
        self.train_patches = self.fd.generate_descriptors(self.images, quiet=False)  # torch.float32
        self.train_np = self.train_patches.cpu().numpy().astype(np.float32, copy=False)
        self._build_channels_and_index()

    @staticmethod
    def _measure_cdist_cols(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (D, La), b: (D, Lb) -> dist (La, Lb)
        return torch.cdist(a.permute(1,0), b.permute(1,0))

    def _mutual_assoc(self, tgt: torch.Tensor, src: torch.Tensor,
                      tgt_idx: int, src_idx: int) -> np.ndarray:
        """
        Associazione reciproca (mutual NN) fra patch di due immagini.
        Ritorna array (Ltgt, 5) con [idx_tgt, idx_src, dist, tgt_img, src_img] o inf se non reciproco.
        """
        device = tgt.device
        La = tgt.size(1); Lb = src.size(1)
        block = 1024
        # min su colonne e righe
        src_zero_min = torch.full((La,), float('inf'), device=device)
        src_zero_arg = torch.zeros((La,), device=device, dtype=torch.long)
        tgt_one_min  = torch.full((Lb,), float('inf'), device=device)
        tgt_one_arg  = torch.zeros((Lb,), device=device, dtype=torch.long)

        for xs in range(0, Lb, block):
            for yt in range(0, La, block):
                d = self._measure_cdist_cols(src[:, xs:xs+block], tgt[:, yt:yt+block])  # (Lb', La')
                mins0, args0 = torch.min(d, dim=0)  # min per tgt-col
                cond0 = src_zero_min[yt:yt+mins0.numel()] >= mins0
                src_zero_arg[yt:yt+mins0.numel()] = torch.where(cond0, (args0 + xs), src_zero_arg[yt:yt+mins0.numel()])
                src_zero_min[yt:yt+mins0.numel()] = torch.minimum(src_zero_min[yt:yt+mins0.numel()], mins0)

                mins1, args1 = torch.min(d, dim=1)  # min per src-row
                cond1 = tgt_one_min[xs:xs+mins1.numel()] >= mins1
                tgt_one_arg[xs:xs+mins1.numel()] = torch.where(cond1, (args1 + yt), tgt_one_arg[xs:xs+mins1.numel()])
                tgt_one_min[xs:xs+mins1.numel()] = torch.minimum(tgt_one_min[xs:xs+mins1.numel()], mins1)

        src_idx_arr = src_zero_arg.long().cpu().numpy()
        tgt_idx_arr = tgt_one_arg.long().cpu().numpy()
        mins_tgt = tgt_one_min.detach().cpu().numpy().astype(np.float32)

        out = np.ones((tgt_idx_arr.shape[0], 5), dtype=np.float32) * np.inf
        for x in range(tgt_idx_arr.shape[0]):
            if src_idx_arr[tgt_idx_arr[x]] == x:
                out[x] = np.array([x, tgt_idx_arr[x], mins_tgt[x], tgt_idx, src_idx], dtype=np.float32)
            else:
                out[x] = np.array([np.inf, np.inf, mins_tgt[x], np.inf, np.inf], dtype=np.float32)
        return out

    def _build_channels_and_index(self):
        """
        Costruisce canali con mutua associazione seed→compare e fa pruning (std sferica).
        Poi costruisce l'indice FAISS sui "nominal points".
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N, D, L = self.train_patches.shape

        assoc = np.ones((self.assoc_depth, N, L, 5), dtype=np.float32) * np.inf

        # step 1: associazione reciproca seed vs immagini successive
        for s in tqdm(range(min(self.assoc_depth, N)), ncols=100, desc="Associate To Channels"):
            tgt = self.train_patches[s].to(device=device, dtype=torch.float32)  # (D,L)
            for j in range(s + 1, N):
                src = self.train_patches[j].to(device=device, dtype=torch.float32)
                assoc[s, j] = self._mutual_assoc(tgt, src, s, j)

        # step 2: scegli per (N,L) il seed con distanza minima lungo S (PATCH APPLICATA)
        S, Ntot, Ltot, _ = assoc.shape
        assert Ntot == N and Ltot == L
        mins_idx = np.argmin(assoc[:, :, :, 2], axis=0).astype(np.int64)  # (N, L)
        n_idx = np.arange(N)[:, None]                                     # (N,1)
        l_idx = np.arange(L)[None, :]                                     # (1,L)
        assoc_best = assoc[mins_idx, n_idx, l_idx, :]                      # (N, L, 5)

        # step 3: crea canali
        channels: Dict[str, List] = {}
        for img_i in tqdm(range(N), ncols=100, desc="Create Channels"):
            for p in range(L):
                row = assoc_best[img_i, p]
                if row[0] < np.inf:  # reciproco
                    cname = f"{int(row[0])}_{int(row[3])}"  # patchId_seedImg
                    if cname not in channels:
                        # seed center
                        seed_patch_vec = self.train_np[int(row[3]), :, int(row[0])]
                        channels[cname] = [[seed_patch_vec, int(row[3]), int(row[0])]]
                    # add src
                    src_patch_vec = self.train_np[int(row[4]), :, int(row[1])]
                    channels[cname].append([src_patch_vec, int(row[4]), int(row[1])])

        # step 4: pruning canali (std sferica rispetto al centro medio)
        nominal_points = []
        for cname in tqdm(list(channels.keys()), ncols=100, desc="Filter Channels"):
            clist = channels[cname]
            if len(clist) > self.min_channel_length:
                cpatch = np.array([c[0] for c in clist], dtype=np.float32)  # (m, D)
                mean = np.mean(cpatch, axis=0, dtype=np.float32)
                dists = np.sqrt(np.sum((cpatch - mean) ** 2, axis=1, dtype=np.float32)).astype(np.float32)
                std = float(np.std(dists, axis=0, dtype=np.float32))
                keep = [c for c, dist in zip(clist, dists) if dist < (self.max_channel_std * std + 1e-12)]
                if len(keep) > self.min_channel_length:
                    nominal_points += [np.asarray(c[0], dtype=np.float32) for c in keep]

        if len(nominal_points) == 0:
            raise RuntimeError("Dopo il pruning non restano nominal points. Aumenta ASSOC_DEPTH o MAX_CHANNEL_STD.")

        base_np = np.array(nominal_points, dtype=np.float32, order="C")  # (Nbase, D)
        d = int(base_np.shape[1])
        if torch.cuda.is_available():
            try:
                self.nn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d, faiss.GpuIndexFlatConfig())
            except Exception:
                self.nn = faiss.IndexFlatL2(d)
        else:
            self.nn = faiss.IndexFlatL2(d)
        self.nn.add(base_np.astype(np.float32, copy=False))

    def predict(self, test_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Ritorna score maps (H_img, W_img) per ciascuna immagine di test.
        """
        test_desc = self.fd.generate_descriptors(test_images, quiet=False)  # (N, D, L)
        outs = []
        for i in tqdm(range(test_desc.size(0)), ncols=100, desc="Predicting On Images"):
            q = torch.permute(test_desc[i], (1, 0)).contiguous().cpu().numpy().astype(np.float32)  # (L, D)
            # kNN repo-faithful: k=1, senza L2-normalization
            dist, _ = self.nn.search(q, KNN_K)  # (L,1)
            dist1 = dist[:, 0].astype(np.float32, copy=False)

            L = dist1.shape[0]
            side = int(np.sqrt(L))
            assert side * side == L, f"L={L} non quadrato; controlla allineamento griglia patch."
            dist2d = dist1.reshape(side, side)

            # Bilinear upsample fino alla risoluzione dell'immagine di input
            Himg, Wimg = test_images[i].shape[:2]
            dist2d_t = torch.from_numpy(dist2d)[None, None, ...].to(dtype=torch.float32)
            dist_up = F.interpolate(dist2d_t, size=(Himg, Wimg), mode="bilinear", align_corners=False)[0,0].cpu().numpy()

            outs.append(gaussian_filter(dist_up, self.filter_size).astype(np.float32, copy=False))
        return outs

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

    # backbone e nodi (allineati alla repo; livelli da blocchi 1-4)
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

    # InReaCh (repo-faithful)
    inreach = InReaCh(
        images=train_imgs,
        model=model,
        return_nodes_keys_in_order=list(return_nodes.values()),  # <<-- OUTPUT-keys (FIX)
        assoc_depth=ASSOC_DEPTH,
        min_channel_length=MIN_CHANNEL_LENGTH,
        max_channel_std=MAX_CHANNEL_STD,
        filter_size=FILTER_SIZE,
        use_positional_embeddings=True,
        pos_embed_dim=64
    )

    # Val set -> liste
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

    # image-level score: max
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
