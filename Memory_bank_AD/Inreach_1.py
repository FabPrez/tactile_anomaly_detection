# filename: InReach.py
#

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinismo CUDA

import math, random, time
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from tqdm import tqdm
import tqdm as tq
import cv2

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.ndimage import gaussian_filter

# --- tue utility (progetto) ---
from data_loader import build_ad_datasets, make_loaders
from view_utils import show_dataset_images, show_validation_grid_from_loader
from ad_analysis import run_pixel_level_evaluation, print_pixel_report

# --- FAISS ---
try:
    import faiss
except Exception as e:
    raise ImportError(
        "InReaCh richiede FAISS. Installa: pip install faiss-cpu (oppure conda: faiss-cpu/faiss-gpu)."
    ) from e

torch.set_default_dtype(torch.float32)

# ---------------- CONFIG ----------------
METHOD = "INREACH_OFFICIAL"
CODICE_PEZZO = "PZ2"

TRAIN_POSITIONS = ["pos5"]
VAL_GOOD_PER_POS = 20
VAL_GOOD_SCOPE   = ["pos5"]
VAL_FAULT_SCOPE  = ["pos5"]
GOOD_FRACTION    = 0.1

IMG_SIZE  = 224
SEED      = 42

# InReaCh (repo)
ASSOC_DEPTH        = 10
MIN_CHANNEL_LENGTH = 3
MAX_CHANNEL_STD    = 5.0
FILTER_SIZE        = 13

# k-NN repo-faithful
KNN_K = 1

# Positional embedding gate (stile repo)
POS_EMBED_THRESH_DEFAULT = 1000.0
POS_EMBED_WEIGHT_ON      = 5.0     # fattore di scala dei due canali (y,x) quando gate=ON

# Visual
VIS_VALID_DATASET               = False
VIS_PREDICTION_ON_VALID_DATASET = True

# -------------------- Inlined "utils.py" (repo) --------------------

def measure_distances(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Euclidean tra colonne: features_a: (D, Na), features_b: (D, Nb)
    Restituisce matrice distanze di shape (Na, Nb) (coerente con repo: s_len x t_len).
    """
    return torch.cdist(torch.permute(features_a, (1, 0)), torch.permute(features_b, (1, 0)))

def super_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

def test_for_positional_class_transpose(imgs: List[np.ndarray]) -> float:
    """
    Test repo: media immagini vs immagine trasposta. Se differenza quadratica media è bassa,
    la classe è "positional" (PE ON).
    """
    average = np.mean(np.array(imgs), axis=0, keepdims=False)
    average_f = np.transpose(average, (1, 0, 2))
    return float(np.mean(np.square(average.astype(np.float16) - average_f.astype(np.float16))))

def test_for_positional_class_flips(imgs: List[np.ndarray]) -> float:
    """
    Variante col flip (non usata nel gate sotto, lasciata per allineamento repo).
    """
    average = np.mean(np.array(imgs), axis=0, keepdims=False)
    average_f = np.flip(np.flip(average, (0)), (1))
    return float(np.mean(np.square(average.astype(np.float16) - average_f.astype(np.float16))))

def align_images(seed: np.ndarray,
                 images: List[np.ndarray],
                 masks: Optional[List[np.ndarray]],
                 quite: bool = True) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    """
    Allineamento rigido per rotazione: prova 0..359°, sceglie la rotazione che minimizza MSE su una finestra centrale.
    """
    c_x = seed.shape[0] // 2
    c_y = seed.shape[1] // 2
    image_size = (seed.shape[1], seed.shape[0])  # cv2 usa (W,H)
    r_mat = [cv2.getRotationMatrix2D((c_x, c_y), x, 1.0) for x in range(360)]

    proposed_data_corrupted_images = []
    proposed_used_test_masks = []

    test_img = seed.astype(np.float16)[c_x - (c_x // 2): c_x + (c_x // 2),
                                       c_y - (c_y // 2): c_y + (c_y // 2)]

    for k, image in enumerate(tq.tqdm(images, ncols=100, desc='Rotating', disable=quite)):
        rotation_ideal = []
        for x in range(360):
            candidate_full = cv2.warpAffine(image, r_mat[x], image_size).astype(np.float16)
            candidate = candidate_full[c_x - (c_x // 2): c_x + (c_x // 2),
                                       c_y - (c_y // 2): c_y + (c_y // 2)]
            rotation_ideal.append(np.mean(np.square(test_img - candidate)))
        best = int(np.argmin(rotation_ideal))
        proposed_data_corrupted_images.append(cv2.warpAffine(image, r_mat[best], image_size))
        if masks is not None:
            masks_rounded = cv2.warpAffine(masks[k], r_mat[best], image_size)
            masks_rounded[masks_rounded > 128] = 255
            masks_rounded[masks_rounded <= 128] = 0
        else:
            masks_rounded = None
        proposed_used_test_masks.append(masks_rounded)

    return proposed_data_corrupted_images, proposed_used_test_masks

def positional_test_and_alignment(images: List[np.ndarray],
                                  threashold: float,
                                  masks: Optional[List[np.ndarray]] = None,
                                  align: bool = True,
                                  quite: bool = True) -> Tuple[bool, List[np.ndarray], Optional[List[np.ndarray]], bool]:
    """
    Gate "ufficiale" della repo:
      - se test_for_positional_class_transpose(images) < soglia -> classe positional.
      - Se align=True, tenta un allineamento e ri-testa: se dopo l'allineamento il test NON passa,
        allora la posizione "conta" (PE ON) e ritorna (True, imgs_allineate, masks_allineate, True).
      - Altrimenti, se il test passa già o continua a passare dopo align, ritorna (False, imgs, masks, False).
    """
    if test_for_positional_class_transpose(images) < threashold:
        if align:
            proposed_images, proposed_masks = align_images(images[0], images, masks, quite=quite)
            if test_for_positional_class_transpose(proposed_images) < threashold:
                return False, images, masks, False
            else:
                return True, proposed_images, proposed_masks, True
        return False, images, masks, False
    return True, images, masks, False

# ---------------- MODEL LOADER (repo-like torch.hub v0.10.0) ----------------
from torchvision.models.feature_extraction import create_feature_extractor

def load_wide_resnet_50(return_nodes: Dict[str, str] = None,
                        verbose: bool = False,
                        size=(3, 224, 224)):
    from torchvision.models import Wide_ResNet50_2_Weights
    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'wide_resnet50_2',
        Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        force_reload=False,
        verbose=False
    )
    if return_nodes is not None:
        model = create_feature_extractor(model, return_nodes=return_nodes)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

# ---------------- FeatureDescriptors (repo) con NO ImageNet normalize ----------------
import abc

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            self.preprocessing_modules.append(MeanMapper(output_dim))

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        # Evita .cuda() forzato: mantieni device di 'features'
        features = features.to(dtype=torch.float32)
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

class PatchMakerFD:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = 1 if stride is None else stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)  # [B, C*ps*ps, L]
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)  # [B, L, C, ps, ps]

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)

class Feautre_Descriptor(abc.ABC):
    """
    Versione repo con UNA modifica: image_net_norm() rimpiazzata da scaling 0..1 (NO ImageNet normalize).
    """
    def __init__(self,
                 model : torch.nn.Module,
                 image_size: tuple = (224,224,3),
                 flatten_output: bool = True,
                 positional_embeddings: float = 5.0,
                 pretrain_embed_dimension: int = 1024,
                 target_embed_dimension: int = 1024,
                 agg_stride: int = 1,
                 agg_size: int = 3):
        self.model = model
        self.flatten_output = flatten_output
        self.positional_embeddings = float(positional_embeddings)
        self.pretrain_embed_dimension = pretrain_embed_dimension
        self.target_embed_dimension = target_embed_dimension

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_image = torch.from_numpy(
            np.transpose(np.zeros(shape=(1, *image_size), dtype=np.float32), axes=[0, 3, 1, 2])
        ).to(device)
        with torch.no_grad():
            features = self.model(test_image)
        self.feature_size = [(features[layer].size(2), features[layer].size(3)) for layer in features.keys()]

        self.patch_maker = PatchMakerFD(patchsize=agg_size, stride=agg_stride)
        self.agg_preprocessing = Preprocessing(
            [features[layer].size(1) for layer in features.keys()],
            self.pretrain_embed_dimension
        )
        self.pre_adapt_aggregator = Aggregator(target_dim=self.target_embed_dimension)

    def image_net_norm(self, image: np.ndarray) -> torch.Tensor:
        """
        Sostituito: SOLO scaling 0..1 (NO mean/std ImageNet).
        Restituisce tensor float32 in NCHW.
        """
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image, axes=[0, 3, 1, 2])  # NCHW
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        return torch.tensor(image, dtype=torch.float32)

    def generate_descriptors(self, images: List[np.ndarray], quite: bool = False):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            output = []
            for _, image in enumerate(tqdm(images, ncols=100,
                                           desc='Gen Feature Descriptors',
                                           disable=quite)):
                x = self.image_net_norm(image).to(device)
                feats_dict = self.model(x)
                features_list = [feats_dict[layer] for layer in feats_dict.keys()]  # in ordine

                # Patchify per ogni layer + info (Hi, Wi)
                features_pw = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_list]
                patch_shapes = [x[1] for x in features_pw]
                features_pw  = [x[0] for x in features_pw]  # [B, L, C, ps, ps] per layer

                ref_num_patches = patch_shapes[0]  # [Href, Wref]
                # Riallinea ogni layer alla griglia ref via bilinear
                for i in range(1, len(features_pw)):
                    _features = features_pw[i]
                    patch_dims = patch_shapes[i]  # [Hi, Wi]

                    _features = _features.reshape(
                        _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                    )  # [B, Hi, Wi, C, ps, ps]
                    _features = _features.permute(0, -3, -2, -1, 1, 2)  # [B, C, ps, ps, Hi, Wi]
                    perm_base_shape = _features.shape
                    _features = _features.reshape(-1, *_features.shape[-2:])  # [B*C*ps*ps, Hi, Wi]
                    _features = F.interpolate(
                        _features.unsqueeze(1),
                        size=(ref_num_patches[0], ref_num_patches[1]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)  # [B*C*ps*ps, Href, Wref]
                    _features = _features.reshape(
                        *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                    )  # [B, C, ps, ps, Href, Wref]
                    _features = _features.permute(0, -2, -1, 1, 2, 3)  # [B, Href, Wref, C, ps, ps]
                    _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                    features_pw[i] = _features

                # appiattisci per layer a (B*Lref, C, ps, ps)
                features_pw = [x.reshape(-1, *x.shape[-3:]) for x in features_pw]

                # Preprocessing + Aggregator
                mapped = self.agg_preprocessing(features_pw)  # [B*Lref per layer -> aggregato per layer]
                pooled = self.pre_adapt_aggregator(mapped)    # [B*Lref, target_dim]
                # Rimetti a [Href, Wref, D]
                pooled = torch.reshape(
                    pooled,
                    (*self.feature_size[0], self.target_embed_dimension)
                )  # (Href, Wref, D)
                output.append(pooled.unsqueeze(0).cpu())  # Store su CPU per risparmiare GPU

                del feats_dict, features_list, features_pw, mapped, pooled, x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            output = torch.cat(output, axis=0)  # [N, Href, Wref, D]
            output = torch.permute(output, (0, 3, 1, 2))  # [N, D, Href, Wref]
            shape = output.size()

            # === Positional embeddings (repo-style: due canali y/x scalati) ===
            if self.positional_embeddings > 0:
                with torch.no_grad():
                    # canale Y: rampa verticale normalizzata e scalata
                    pos_y_vals = torch.arange(0, shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(3)
                    pos_y_vals = torch.mul(pos_y_vals, self.positional_embeddings / shape[2])
                    pos_y = pos_y_vals.repeat(shape[0], 1, 1, shape[3])  # [N,1,H,W]

                    # canale X: rampa orizzontale normalizzata e scalata
                    pos_x_vals = torch.arange(0, shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(2)
                    pos_x_vals = torch.mul(pos_x_vals, self.positional_embeddings / shape[3])
                    pos_x = pos_x_vals.repeat(shape[0], 1, shape[2], 1)  # [N,1,H,W]

                    positions = torch.cat([pos_y, pos_x], dim=1)  # [N,2,H,W]
                    output = torch.cat([output, positions], dim=1)

            if self.flatten_output:
                shape = output.size()
                output = torch.reshape(output, (shape[0], shape[1], shape[2] * shape[3]))  # [N, D(+2), L]

        return output

# ---------------- InReaCh (repo) con gate inline e NO ImageNet normalize ----------------
class InReaCh:
    def __init__(self, 
                 images: List[np.ndarray], 
                 model : torch.nn.Module,
                 assoc_depth: int = 10,
                 min_channel_length: int = 3,
                 max_channel_std: float = 5.0,
                 masks: Optional[List[np.ndarray]] = None, 
                 quite: bool = False,
                 pos_embed_thresh: float = 600.0,
                 pos_embed_weight: float = 5.0,
                 filter_size: float = 5,
                 **kwargs) -> None:
        
        self.quite = quite
        self.images = images
        self.masks = masks
        self.image_size = tuple(images[0].shape)
        self.model = model
        self.assoc_depth = assoc_depth
        self.filter_size = int(filter_size)
        self.min_channel_length = min_channel_length
        self.max_channel_std = max_channel_std

        # Gate PE e allineamento ATTIVO (align=True) come flusso ufficiale
        self.pos_embed_flag, self.images, self.masks, self.aligment_flag = positional_test_and_alignment(
            images=self.images,
            threashold=pos_embed_thresh,
            masks=self.masks,
            align=True,            # repo-like
            quite=self.quite
        )
        # 🔧 Patch repo-faithful:
        # se l'utente non ha passato maschere a __init__ (masks=None),
        # ignoriamo eventuali liste di None restituite dall'allineamento.
        if masks is None:
            self.masks = None

        self.pos_embed_weight = float(pos_embed_weight) if self.pos_embed_flag else 0.0

        # Feature Extraction (NO ImageNet normalize)
        self.fd_gen = Feautre_Descriptor(
            model=model,
            image_size=self.image_size,
            positional_embeddings=self.pos_embed_weight,
            **kwargs
        )
        self.patches = self.fd_gen.generate_descriptors(self.images, quite=self.quite)  # [N, D, L]
        self.cpu_patches = self.patches.cpu().numpy().astype(np.float32, copy=False)
        
        # Se ho maschere, prepara contatori precision/recall per canali
        if self.masks is not None:
            L = self.cpu_patches.shape[2]
            side = int(np.sqrt(L))
            self.patch_shape = (side, side)
            self.scale = self.masks[0].shape[0] // self.patch_shape[0]  # assume quadrate
            self.tp = 0
            self.fp = 0
            # maschere a 3 canali -> conta non-neri
            self.negatives = (np.count_nonzero(self.masks) // 3) // max(1, self.scale)
            self.positives = L * self.cpu_patches.shape[0] - self.negatives
            self.max_label = int(np.max(np.array(self.masks)))

        # Costruisci i canali e l'indice FAISS
        self.gen_channels(self.quite)

    def gen_assoc(self, targets: torch.Tensor, 
                         sources: torch.Tensor, 
                         target_img_index: int, 
                         source_img_indexs: int):
        """
        Mutual association tra 'targets' e 'sources' (entrambi [D, L]).
        Restituisce array [L_targets, 5] con [idx_tgt, idx_src, dist_min, target_img_idx, source_img_idx] per match reciproci.
        """
        device = targets.device
        t_len = targets.size(1)
        s_len = sources.size(1)

        sources_zero_axis_min   = torch.full((t_len,), float('inf'), device=device)
        sources_zero_axis_index = torch.zeros((t_len,), device=device)
        targets_ones_axis_min   = torch.full((s_len,), float('inf'), device=device)
        targets_ones_axis_index = torch.zeros((s_len,), device=device)

        # Gestione memoria: blocca in base a memoria disponibile (solo se CUDA disponibile)
        if torch.cuda.is_available():
            aval_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            max_side = int(np.floor(np.sqrt(max(1, aval_mem // 32))))
            max_side = max(1, max_side)
        else:
            max_side = 1024  # fallback CPU "ragionevole"

        for x in range(int(np.ceil(s_len / max_side))):
            for y in range(int(np.ceil(t_len / max_side))):
                s_slice = sources[:, x * max_side: min((x + 1) * max_side, s_len)]
                t_slice = targets[:, y * max_side: min((y + 1) * max_side, t_len)]

                distances = measure_distances(s_slice, t_slice)  # shape (s_sub, t_sub)

                mins, args = torch.min(distances, dim=0)  # min over sources per each target
                y0 = y * max_side
                y1 = y0 + mins.numel()
                cond0 = sources_zero_axis_min[y0:y1] >= mins
                sources_zero_axis_index[y0:y1] = torch.where(cond0, args.to(device) + x * max_side,
                                                             sources_zero_axis_index[y0:y1])
                sources_zero_axis_min[y0:y1] = torch.minimum(sources_zero_axis_min[y0:y1], mins)

                mins, args = torch.min(distances, dim=1)  # min over targets per each source
                x0 = x * max_side
                x1 = x0 + mins.numel()
                cond1 = targets_ones_axis_min[x0:x1] >= mins
                targets_ones_axis_index[x0:x1] = torch.where(cond1, args.to(device) + y * max_side,
                                                             targets_ones_axis_index[x0:x1])
                targets_ones_axis_min[x0:x1] = torch.minimum(targets_ones_axis_min[x0:x1], mins)

        sources_indexs = sources_zero_axis_index.detach().cpu().numpy().astype(int)
        targets_indexs = targets_ones_axis_index.detach().cpu().numpy().astype(int)

        assoc = np.ones((targets_indexs.shape[0], 5), dtype=np.float32) * np.inf
        t1 = targets_ones_axis_min.detach().cpu().numpy().astype(np.float32)
        for x in range(targets_indexs.shape[0]):
            if sources_indexs[targets_indexs[x]] == x:
                assoc[x] = np.array([x, targets_indexs[x], t1[x], target_img_index, source_img_indexs], dtype=np.float32)
            else:
                assoc[x] = np.array([np.inf, np.inf, t1[x], np.inf, np.inf], dtype=np.float32)

        return assoc

    def get_precision_recall(self):
        if self.masks is not None:
            precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
            recall = self.tp / self.positives if self.positives > 0 else 0.0
            return precision, recall
        else:
            return -1.0, -1.0

    def precision_recall(self, patches: List[list]):
        """
        Aggiorna TP/FP valutando, per ciascun patch selezionato nel canale, se la corrispondente cella di mask è 'buona' (0) o anomala.
        """
        if self.masks is not None:
            for x in range(len(patches)):
                index = np.unravel_index(patches[x][2], shape=self.patch_shape)
                if np.average(self.masks[patches[x][1]][
                    index[0]*self.scale:(index[0]+1)*self.scale,
                    index[1]*self.scale:(index[1]+1)*self.scale, :
                ]) == 0:
                    self.tp += 1
                else:
                    self.fp += 1

    def gen_channels(self, quite: bool = False):
        # Colleziona associazioni
        assoc = np.ones((self.assoc_depth, self.patches.size(0), self.patches.size(2), 5), dtype=np.float32) * np.inf
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed_index in tq.tqdm(range(min(self.assoc_depth, self.patches.size(0))), ncols=100,
                                  desc='Associate To Channels', disable=quite):
            gpu_seeds = self.patches[seed_index].to(device)
            for compare_index in range(seed_index + 1, self.patches.size(0)):
                assoc[seed_index, compare_index] = self.gen_assoc(
                    gpu_seeds, self.patches[compare_index].to(device),
                    seed_index, compare_index
                )

        # Per ogni patch, scegli il seed con distanza minima lungo S
        assoc_best = np.take_along_axis(
            assoc,
            np.expand_dims(assoc[:, :, :, 2], axis=3).argmin(axis=0)[None],
            axis=0
        )[0]  # shape: (N, L, 5)
        # Appiattisci (N * L, 5)
        assoc_flat = np.resize(assoc_best, (assoc_best.shape[0] * assoc_best.shape[1], assoc_best.shape[2]))

        # Crea canali
        channels: Dict[str, List] = {}
        for p_index in tq.tqdm(range(assoc_flat.shape[0]), ncols=100, desc='Create Channels', disable=quite):
            if assoc_flat[p_index, 0] < np.inf:
                channel_name = f"{int(assoc_flat[p_index, 0])}_{int(assoc_flat[p_index, 3])}"
                if channel_name in channels:
                    channels[channel_name].append([
                        self.cpu_patches[int(assoc_flat[p_index, 4]), :, int(assoc_flat[p_index, 1])],
                        int(assoc_flat[p_index, 4]),
                        int(assoc_flat[p_index, 1])
                    ])
                else:
                    channels[channel_name] = [[
                        self.cpu_patches[int(assoc_flat[p_index, 3]), :, int(assoc_flat[p_index, 0])],
                        int(assoc_flat[p_index, 3]),
                        int(assoc_flat[p_index, 0])
                    ]]
                    channels[channel_name].append([
                        self.cpu_patches[int(assoc_flat[p_index, 4]), :, int(assoc_flat[p_index, 1])],
                        int(assoc_flat[p_index, 4]),
                        int(assoc_flat[p_index, 1])
                    ])

        # Indice FAISS
        nominal_points = []
        for channel_name in tq.tqdm(list(channels.keys()), ncols=100, desc='Filter Channels', disable=quite):
            if len(channels[channel_name]) > self.min_channel_length:
                c_patches = np.array([patch[0] for patch in channels[channel_name]], dtype=np.float32)
                mean = np.mean(c_patches, axis=0, dtype=np.float32)
                std = float(np.std(np.sqrt(np.sum((c_patches - mean) ** 2, axis=1, dtype=np.float32)), axis=0))
                new_centers = [center for center in channels[channel_name]
                               if np.sqrt(np.sum((mean - center[0]) ** 2)) < self.max_channel_std * std]
                if len(new_centers) > self.min_channel_length:
                    channels[channel_name] = new_centers
                    self.precision_recall(new_centers)
                    nominal_points += [c[0].astype(np.float32, copy=False) for c in new_centers]
                else:
                    del channels[channel_name]
            else:
                del channels[channel_name]

        if len(nominal_points) == 0:
            raise RuntimeError("Dopo il pruning non restano nominal points. Aumenta ASSOC_DEPTH o MAX_CHANNEL_STD.")

        base_np = np.ascontiguousarray(np.array(nominal_points, dtype=np.float32))  # (Nbase, D)
        d = int(base_np.shape[1])
        if torch.cuda.is_available():
            try:
                self.nn_object = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d, faiss.GpuIndexFlatConfig())
            except Exception:
                self.nn_object = faiss.IndexFlatL2(d)
        else:
            self.nn_object = faiss.IndexFlatL2(d)
        self.nn_object.add(base_np)

    def predict(self, t_images: List[np.ndarray], 
                t_masks: Optional[List[np.ndarray]] = None, 
                quite: bool = False):
        if self.aligment_flag:
            t_images, t_masks = align_images(self.images[0], t_images, t_masks, quite=quite)

        start = time.time()
        t_patches = self.fd_gen.generate_descriptors(t_images, quite=quite)  # [N, D, L]
        
        scores = []
        for test_img_index in tq.tqdm(range(t_patches.size(0)), ncols=100,
                                      desc='Predicting On Images', disable=quite):
            q = torch.permute(t_patches[test_img_index], (1, 0)).contiguous().cpu().numpy().astype(np.float32)  # (L, D)
            dist, ind = self.nn_object.search(q, KNN_K)
            dist = dist[:, 0]
            side = int(np.sqrt(dist.shape[0]))
            dist2d = np.resize(dist, new_shape=(side, side))
            # === Upsampling stile repo: stesso fattore su H e W ===
            rep = max(1, t_images[0].shape[0] // dist2d.shape[0])
            dist_up = dist2d.repeat(rep, axis=0).repeat(rep, axis=1)
            scores.append(gaussian_filter(dist_up, self.filter_size))

        if not quite:
            print('TIME TO COMPLETE all predictions', abs(start - time.time()))
        
        return scores, t_masks
         
    def test(self,  t_images: List[np.ndarray], 
                    t_masks: List[np.ndarray] = None, 
                    quite: bool = False):
        
        scores, t_masks = self.predict(t_images, t_masks=t_masks, quite=quite)
        if t_masks is None:
            raise ValueError("Per la valutazione pixel-level servono le maschere di verità a terra.")

        t_masks_bin = [(mask[:, :, 0] / 255.).astype(int) for mask in t_masks]

        img_scores = [float(np.max(score)) for score in scores] 
        img_masks  = [int(np.max(mask)) for mask in t_masks_bin]

        scores_flat  = np.array(scores, dtype=object)
        scores_flat  = np.concatenate([s.flatten() for s in scores_flat]).astype(np.float32)
        masks_flat   = np.concatenate([m.flatten() for m in t_masks_bin]).astype(np.int32)

        pxl_auroc = roc_auc_score(masks_flat, scores_flat)
        img_auroc = roc_auc_score(img_masks, img_scores)
        p, r = self.get_precision_recall()

        return pxl_auroc, img_auroc, p, r

# ---------------- MAIN (integrazione con le tue utility) ----------------
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

    # Nodi (repo-like)
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

    # InReaCh con PE-gate inline e PE lineari; NO ImageNet normalize
    inreach = InReaCh(
        images=train_imgs,
        model=model,
        assoc_depth=ASSOC_DEPTH,
        min_channel_length=MIN_CHANNEL_LENGTH,
        max_channel_std=MAX_CHANNEL_STD,
        filter_size=FILTER_SIZE,
        pos_embed_thresh=POS_EMBED_THRESH_DEFAULT,
        pos_embed_weight=POS_EMBED_WEIGHT_ON,
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
    score_maps, val_masks_aligned = inreach.predict(val_imgs, t_masks=val_masks, quite=False)

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
