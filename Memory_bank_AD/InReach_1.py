# InReach_official_min.py — InReaCh ufficiale adattato al tuo dataset (float32 end-to-end, senza cv2)
# Dipendenze chiave: torch, torchvision, faiss-cpu, scipy, scikit-learn
# Usa le tue utility: data_loader (build_ad_datasets, make_loaders), ad_analysis, view_utils

import os, math, random, time
from typing import List, Dict, Tuple
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

# --- FAISS (repo ufficiale usa FAISS per la ricerca NN) ---
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
MAX_CHANNEL_STD    = 5.0
FILTER_SIZE        = 13

# Visual
VIS_VALID_DATASET               = False
VIS_PREDICTION_ON_VALID_DATASET = True


# ---------------- Utils (repo: utils.py) ----------------
def super_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

def measure_distances(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    # repo: distances = torch.cdist(torch.permute(a,[1,0]), torch.permute(b,[1,0]))
    # assicurati che siano float32 sullo stesso device
    if features_a.dtype != torch.float32:
        features_a = features_a.to(torch.float32)
    if features_b.dtype != torch.float32:
        features_b = features_b.to(torch.float32)
    return torch.cdist(torch.permute(features_a,[1,0]), torch.permute(features_b,[1,0]))


# ---------------- MODEL (repo: model.py) ----------------
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

def load_wide_resnet_50(return_nodes: Dict[str, str] = None, verbose: bool = False, size=(3, 224, 224)):
    # nella repo usano torch.hub su v0.10.0; qui facciamo fallback sicuro
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


# ---------------- Feature Descriptors (repo: features_descriptors.py) ----------------
import abc

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # input atteso: (L, F) — per L patch, F feature
        x = features.to(torch.float32).reshape(len(features), 1, -1)  # (L,1,F)
        x = F.adaptive_avg_pool1d(x, self.preprocessing_dim).squeeze(1)  # (L,pre_dim)
        return x.to(torch.float32)

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.preprocessing_modules = torch.nn.ModuleList([MeanMapper(output_dim) for _ in input_dims])
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features: lista di T_i con shape (L, F_i)
        outs = []
        for module, feat in zip(self.preprocessing_modules, features):
            outs.append(module(feat))  # (L, pre_dim)
        # stack su asse livelli → (L, n_levels, pre_dim)
        return torch.stack(outs, dim=1).to(torch.float32)

class PatchMaker:
    def __init__(self, patchsize: int, stride: int):
        self.patchsize = patchsize
        self.stride = stride
    def patchify(self, features: torch.Tensor, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded = unfolder(features.to(torch.float32))  # (B, C*ps*ps, L)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded = unfolded.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)  # (B, C, ps, ps, L)
        unfolded = unfolded.permute(0, 4, 1, 2, 3)  # (B, L, C, ps, ps)
        if return_spatial_info:
            return unfolded, number_of_total_patches
        return unfolded

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (L, n_levels, pre_dim) → concat sui livelli e pool a target_dim
        L = features.shape[0]
        x = features.reshape(L, 1, -1)           # (L,1,n_levels*pre_dim)
        x = F.adaptive_avg_pool1d(x, self.target_dim)  # (L,1,target_dim)
        return x.reshape(L, self.target_dim).to(torch.float32)  # (L,D)

class Feautre_Descriptor(abc.ABC):
    def __init__(self,
                 model: torch.nn.Module,
                 image_size: tuple = (224, 224, 3),
                 flatten_output: bool = True,
                 positional_embeddings: float = 0.0,   # disattivo PE per evitare cv2/alignment
                 pretrain_embed_dimension: int = 1024,
                 target_embed_dimension: int = 1024,
                 agg_stride: int = 1,
                 agg_size: int = 3):
        self.model = model
        self.flatten_output = flatten_output
        self.positional_embeddings = positional_embeddings
        self.pretrain_embed_dimension = pretrain_embed_dimension
        self.target_embed_dimension = target_embed_dimension

        device = next(self.model.parameters()).device
        test_image = torch.from_numpy(
            np.transpose(np.zeros(shape=(1, *image_size), dtype=np.float32), axes=[0, 3, 1, 2])
        ).to(device)

        with torch.no_grad():
            feats = self.model(test_image)
        self.feature_size = [(feats[k].size(2), feats[k].size(3)) for k in feats.keys()]
        self.feature_dims = [feats[k].size(1) for k in feats.keys()]

        self.patch_maker = PatchMaker(patchsize=agg_size, stride=agg_stride)
        self.agg_pre = Preprocessing([feats[k].size(1) for k in feats.keys()], self.pretrain_embed_dimension)
        self.aggregator = Aggregator(target_dim=self.target_embed_dimension)

        # porta moduli sullo stesso device del backbone
        self.agg_pre.to(device)
        self.aggregator.to(device)

    def _imagenet_norm(self, img_uint8: np.ndarray, device):
        x = img_uint8.astype(np.float32) / 255.0
        x = (x - np.array([0.456, 0.406, 0.485], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def generate_descriptors(self, images: List[np.ndarray], quite: bool = False) -> torch.Tensor:
        device = next(self.model.parameters()).device
        out_batches = []

        for img in tqdm(images, ncols=100, desc='Gen Feature Descriptors', disable=quite):
            # --- normalize + forward ---
            x = self._imagenet_norm(img, device)
            feats_dict = self.model(x)
            feats = [feats_dict[k].to(torch.float32) for k in feats_dict.keys()]  # livelli

            # --- patchify livelli + riallineo alla stessa griglia (come repo) ---
            feats_p = [self.patch_maker.patchify(f, return_spatial_info=True) for f in feats]
            patch_shapes = [p[1] for p in feats_p]      # [(H0,W0), (H1,W1), ...]
            feats_p = [p[0] for p in feats_p]           # ciascuno: (B, L_i, C, ps, ps)
            ref_hw = patch_shapes[0]                    # (H_ref, W_ref)

            for i in range(1, len(feats_p)):
                _f = feats_p[i]                         # (B, L_i, C, ps, ps)
                Li = patch_shapes[i]
                # porta su griglia ref_hw via bilinear
                _f = _f.reshape(_f.shape[0], Li[0], Li[1], *_f.shape[2:]).permute(0, -3, -2, -1, 1, 2)
                base = _f.shape                         # (B, C, ps, ps, H_i, W_i)
                _f = _f.reshape(-1, *_f.shape[-2:])     # ((B*C*ps*ps), H_i, W_i)
                _f = F.interpolate(_f.unsqueeze(1), size=(ref_hw[0], ref_hw[1]),
                                   mode="bilinear", align_corners=False).squeeze(1)
                _f = _f.reshape(*base[:-2], ref_hw[0], ref_hw[1]).permute(0, -2, -1, 1, 2, 3)
                feats_p[i] = _f.reshape(len(_f), -1, *_f.shape[-3:]).to(torch.float32)  # (B, L_ref, C, ps, ps)

            # --- appiattisci per livello a (L_ref, F_i) ---
            levels_flat = []
            for _f in feats_p:
                _f = _f.squeeze(0).reshape(_f.size(1), -1).to(torch.float32)            # (L_ref, F_i)
                levels_flat.append(_f)

            # --- preprocessing per livello → stack (L_ref, n_levels, pre_dim) ---
            pre = self.agg_pre(levels_flat)                                              # (L_ref, n_levels, pre_dim)

            # --- aggregator lungo i canali/levels → (L_ref, target_dim) ---
            agg = self.aggregator(pre)                                                   # (L_ref, D)

            # --- rimappa: (L_ref, D) → (H_ref, W_ref, D) → (1, D, L_ref) ---
            Hpatch, Wpatch = ref_hw
            L_ref = agg.shape[0]
            assert L_ref == Hpatch * Wpatch, f"Mismatch L={L_ref} vs HxW={Hpatch}x{Wpatch}"

            emb_hw_d = agg.view(Hpatch, Wpatch, self.target_embed_dimension)             # (H,W,D)
            emb_d_hw = emb_hw_d.permute(2, 0, 1).contiguous()                            # (D,H,W)
            emb_d_L  = emb_d_hw.view(self.target_embed_dimension, Hpatch * Wpatch)       # (D,L)

            out_batches.append(emb_d_L.unsqueeze(0).cpu())                               # → (1,D,L)

        # (N, D, L)
        out = torch.cat(out_batches, dim=0).to(torch.float32)
        return out


# ---------------- InReaCh (repo: InReaCh.py) ----------------
class InReaCh:
    def __init__(self,
                 images: List[np.ndarray],
                 model: torch.nn.Module,
                 assoc_depth: int = 10,
                 min_channel_length: int = 3,
                 max_channel_std: float = 5.0,
                 masks: List[np.ndarray] = None,
                 filter_size: int = 13,
                 **kwargs) -> None:

        self.images = images
        self.masks = masks
        self.model = model
        self.assoc_depth = assoc_depth
        self.min_channel_length = min_channel_length
        self.max_channel_std = max_channel_std
        self.filter_size = filter_size

        self.fd_gen = Feautre_Descriptor(
            model=model,
            image_size=images[0].shape,
            flatten_output=True,
            positional_embeddings=0.0,        # niente alignment/PE
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            agg_stride=1,
            agg_size=3
        )
        self.patches = self.fd_gen.generate_descriptors(self.images, quite=False).to(torch.float32)   # (N, D, L)
        self.cpu_patches = self.patches.cpu().numpy().astype(np.float32, copy=False)

        self._build_channels()

    def _mutual_assoc(self, targets: torch.Tensor, sources: torch.Tensor,
                      target_img_idx: int, source_img_idx: int) -> np.ndarray:
        device = targets.device
        targets = targets.to(device=device, dtype=torch.float32)  # (D,L_t)
        sources = sources.to(device=device, dtype=torch.float32)  # (D,L_s)

        t_len = targets.size(1); s_len = sources.size(1)
        big = np.inf

        src_zero_min = torch.full((t_len,), float('inf'), device=device, dtype=torch.float32)
        src_zero_arg = torch.zeros((t_len,), device=device, dtype=torch.float32)
        tgt_one_min  = torch.full((s_len,), float('inf'), device=device, dtype=torch.float32)
        tgt_one_arg  = torch.zeros((s_len,), device=device, dtype=torch.float32)

        block = 1024
        for xs in range(0, s_len, block):
            for yt in range(0, t_len, block):
                d = measure_distances(sources[:, xs:xs+block], targets[:, yt:yt+block])  # (L_s', L_t')
                mins0, args0 = torch.min(d, dim=0)
                cond = src_zero_min[yt:yt+mins0.numel()] >= mins0
                src_zero_arg[yt:yt+mins0.numel()] = torch.where(cond, (args0 + xs).to(torch.float32),
                                                                src_zero_arg[yt:yt+mins0.numel()])
                src_zero_min[yt:yt+mins0.numel()] = torch.minimum(src_zero_min[yt:yt+mins0.numel()], mins0)

                mins1, args1 = torch.min(d, dim=1)
                cond1 = tgt_one_min[xs:xs+mins1.numel()] >= mins1
                tgt_one_arg[xs:xs+mins1.numel()] = torch.where(cond1, (args1 + yt).to(torch.float32),
                                                               tgt_one_arg[xs:xs+mins1.numel()])
                tgt_one_min[xs:xs+mins1.numel()] = torch.minimum(tgt_one_min[xs:xs+mins1.numel()], mins1)

        src_idx = src_zero_arg.long().cpu().numpy()
        tgt_idx = tgt_one_arg.long().cpu().numpy()

        assoc = np.ones((tgt_idx.shape[0], 5), dtype=np.float32) * big
        mins_tgt = tgt_one_min.detach().cpu().numpy().astype(np.float32)
        for x in range(tgt_idx.shape[0]):
            if src_idx[tgt_idx[x]] == x:
                assoc[x] = np.array([x, tgt_idx[x], mins_tgt[x], target_img_idx, source_img_idx], dtype=np.float32)
            else:
                assoc[x] = np.array([big, big, mins_tgt[x], big, big], dtype=np.float32)
        return assoc

    def _build_channels(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assoc = np.ones((self.assoc_depth, self.patches.size(0), self.patches.size(2), 5), dtype=np.float32) * np.inf

        # costruiamo associazioni reciproche seed→compare
        for seed_i in tqdm(range(self.assoc_depth), ncols=100, desc="Associate To Channels"):
            tgt = self.patches[seed_i].to(device=device, dtype=torch.float32)  # (D,L)
            for cmp_i in range(seed_i + 1, self.patches.size(0)):
                src = self.patches[cmp_i].to(device=device, dtype=torch.float32)
                assoc[seed_i, cmp_i] = self._mutual_assoc(tgt, src, seed_i, cmp_i)

        # scegli migliore seed per patch (min dist)
        assoc = np.take_along_axis(assoc, np.expand_dims(assoc[:, :, :, 2], axis=3).argmin(axis=0)[None], axis=0)[0]
        assoc = np.resize(assoc, (assoc.shape[0] * assoc.shape[1], assoc.shape[2]))  # (all_patches, 5)

        # crea canali
        channels: Dict[str, List] = {}
        for p in tqdm(range(assoc.shape[0]), ncols=100, desc="Create Channels"):
            if assoc[p, 0] < np.inf:
                cname = f"{int(assoc[p,0])}_{int(assoc[p,3])}"
                if cname not in channels:
                    channels[cname] = [[self.cpu_patches[int(assoc[p,3]), :, int(assoc[p,0])],
                                        int(assoc[p,3]), int(assoc[p,0])]]
                channels[cname].append([self.cpu_patches[int(assoc[p,4]), :, int(assoc[p,1])],
                                        int(assoc[p,4]), int(assoc[p,1])])

        # filtro canali (repo: std sferica, outlier pruning)
        nominal_points = []
        for cname in tqdm(list(channels.keys()), ncols=100, desc="Filter Channels"):
            if len(channels[cname]) > self.min_channel_length:
                c_patches = np.array([c[0] for c in channels[cname]], dtype=np.float32)
                mean = np.mean(c_patches, axis=0, dtype=np.float32)
                std  = np.std(np.sqrt(np.sum((c_patches - mean) ** 2, axis=1, dtype=np.float32)), axis=0, dtype=np.float32)
                new_centers = [
                    c for c in channels[cname]
                    if float(np.sqrt(np.sum((mean - c[0]) ** 2, dtype=np.float32))) < float(self.max_channel_std * std + 1e-12)
                ]
                if len(new_centers) > self.min_channel_length:
                    nominal_points += [np.asarray(c[0], dtype=np.float32) for c in new_centers]

        if len(nominal_points) == 0:
            raise RuntimeError("Dopo il filtro non restano nominal points. Aumenta ASSOC_DEPTH o MAX_CHANNEL_STD.")

        base_np = np.array(nominal_points, dtype=np.float32, order="C")
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
        t_patches = self.fd_gen.generate_descriptors(test_images, quite=False).to(torch.float32)  # (N, D, L)
        outs = []
        for i in tqdm(range(t_patches.size(0)), ncols=100, desc="Predicting On Images"):
            q = torch.permute(t_patches[i], (1, 0)).contiguous().cpu().numpy().astype(np.float32)  # (L, D)
            dist, _ = self.nn.search(q, 1)  # k=1
            side = int(np.sqrt(dist.shape[0]))
            dist2d = dist[:, 0].reshape(side, side).astype(np.float32, copy=False)
            scale = test_images[0].shape[0] // side
            dist2d = dist2d.repeat(scale, axis=0).repeat(scale, axis=1)
            outs.append(gaussian_filter(dist2d, self.filter_size))
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

    # backbone e nodi (repo: return_nodes)
    return_nodes = {
        'layer1.0.relu_2': 'Level_1',
        'layer1.1.relu_2': 'Level_2',
        'layer1.2.relu_2': 'Level_3',
        'layer2.0.relu_2': 'Level_4',
        'layer2.1.relu_2': 'Level_5',
        'layer2.2.relu_2': 'Level_6',
        'layer2.3.relu_2': 'Level_7',
        'layer3.1.relu_2': 'Level_8',
        'layer3.2.relu_2': 'Level_9',
        'layer3.3.relu_2': 'Level_10',
        'layer3.4.relu_2': 'Level_11',
        'layer3.5.relu_2': 'Level_12',
        'layer4.0.relu_2': 'Level_13',
    }
    model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)

    # InReaCh
    inreach = InReaCh(
        images=train_imgs,
        model=model,
        assoc_depth=ASSOC_DEPTH,
        min_channel_length=MIN_CHANNEL_LENGTH,
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
