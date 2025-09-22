import torch, sys, platform
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import math

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix

# miei pacchetti
from data_loader import get_items, save_split_pickle, load_split_pickle
from view_utils import show_dataset_images, show_validation_grid_from_loader, make_pixel_masks_manual, show_pixel_localization_grid


# ----------------- CONFIG -----------------
CODICE_PEZZO = "PZ1"
POSITION    = "pos1"    # oppure "all"
TOP_K       = 5
IMG_SIZE    = 224       # input ResNet
SEED        = 42
P_PIX       = 99.0      # percentile per soglia pixel-level
GAUSSIAN_SIGMA = 4      # sigma per filtro gaussiano
# ------------------------------------------


# ---------- util ----------
def calc_dist_matrix(x, y):
    """Euclidean distance matrix tra righe di x (n,d) e y (m,d) -> (n,m)."""
    # (puoi sostituire con torch.cdist(x, y))
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


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


def l2norm(x, dim=1, eps=1e-6):
    """Normalizzazione L2 lungo la dimensione 'dim' (cosine-like distance)."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ---------- dataset ----------
class MyDataset(Dataset):
    def __init__(self, images, label_type: str, transform=None):
        """
        images: Sequence[PIL.Image.Image] (RGB)
        label_type: "good" -> 0, "fault" -> 1 (tutte uguali)
        transform: pre-process PIL→PIL (es. Resize, CenterCrop), **senza** ToTensor
        """
        assert label_type in {"good", "fault"}, "label_type deve essere 'good' o 'fault'"
        label_value = 0 if label_type == "good" else 1

        self.data = []
        # costanti normalizzazione ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        for img_pil in images:
            assert isinstance(img_pil, Image.Image), "Le immagini devono essere PIL.Image"
            if transform is not None:
                img_pil = transform(img_pil)  # ancora PIL
            # PIL → Tensor (C,H,W) float32 in [0,1]
            img_t = torch.tensor(np.array(img_pil), dtype=torch.float32).permute(2, 0, 1) / 255.0
            # Normalizzazione ImageNet (fondamentale per backbone pre-addestrato)
            img_t = (img_t - mean) / std
            self.data.append(img_t)

        self.data   = torch.stack(self.data)  # (N, C, H, W)
        self.labels = torch.full((len(self.data),), fill_value=label_value, dtype=torch.long)

        print('data shape', self.data.shape)
        print('label shape', self.labels.shape)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device found:", device)

    # transform deterministica (ordine replicabile) — standard ImageNet
    pre_processing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),  # 224
    ])

    # carico immagini PIL
    good_imgs_pil  = get_items(CODICE_PEZZO, modality="rgb", label="good",  positions=POSITION, return_type="pil")
    fault_imgs_pil = get_items(CODICE_PEZZO, modality="rgb", label="fault", positions=POSITION, return_type="pil")

    # dataset
    good_ds  = MyDataset(good_imgs_pil,  label_type='good',  transform=pre_processing)
    fault_ds = MyDataset(fault_imgs_pil, label_type='fault', transform=pre_processing)

    # split: train (solo good), val (k good + tutte fault)
    n_good, n_fault = len(good_ds), len(fault_ds)
    k = min(n_fault, n_good)

    g = torch.Generator().manual_seed(SEED)
    perm_good  = torch.randperm(n_good,  generator=g) # ordine casuale degli indici delle immagini good
    perm_fault = torch.randperm(n_fault, generator=g) # ordine casuale degli indici delle immagini fault

    val_good_idx   = perm_good[:k].tolist()
    train_good_idx = perm_good[k:].tolist()
    val_fault_idx  = perm_fault[:k].tolist()  # tutte le fault

    train_set     = Subset(good_ds,  train_good_idx)
    val_good_set  = Subset(good_ds,  val_good_idx)
    val_fault_set = Subset(fault_ds, val_fault_idx)
    val_set       = ConcatDataset([val_good_set, val_fault_set])

    print(f"Train: {len(train_set)} good")
    print(f"Val:   {len(val_good_set)} good + {len(val_fault_set)} fault = {len(val_set)}")

    # DataLoader: su Windows evita multiprocess per stabilità (num_workers=0)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)

    # modello + hook
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model   = wide_resnet50_2(weights=weights).to(device)
    model.eval()

    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    test_outputs  = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

    # ====== TRAIN FEATURES (cache) ======
    try:
        train_outputs = load_split_pickle(CODICE_PEZZO, POSITION, split="train")
        print("[cache] Train features caricate da pickle.")
    except FileNotFoundError:
        print("[cache] Nessun pickle train: estraggo feature...")
        for x, y in tqdm(train_loader, desc='| feature extraction | train | custom |'):
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                _ = model(x)
            for k_, v in zip(train_outputs.keys(), outputs):
                train_outputs[k_].append(v.detach())
            outputs = []
        for k_ in train_outputs:
            train_outputs[k_] = torch.cat(train_outputs[k_], dim=0)
        save_split_pickle(train_outputs, CODICE_PEZZO, POSITION, split="train")

    # ====== VAL FEATURES (cache) ======
    try:
        pack         = load_split_pickle(CODICE_PEZZO, POSITION, split="validation")
        test_outputs = pack['features']
        gt_list      = pack['labels']
        print("[cache] Validation features caricate da pickle.")
    except FileNotFoundError:
        print("[cache] Nessun pickle validation: estraggo feature...")
        gt_list = []
        for x, y in tqdm(val_loader, desc='| feature extraction | validation | custom |'):
            gt_list.extend(y.cpu().numpy())
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                _ = model(x)
            for k_, v in zip(test_outputs.keys(), outputs):
                test_outputs[k_].append(v.detach())
            outputs = []
        for k_ in test_outputs:
            test_outputs[k_] = torch.cat(test_outputs[k_], dim=0)
        pack = {'features': test_outputs, 'labels': np.array(gt_list, dtype=np.int64)}
        save_split_pickle(pack, CODICE_PEZZO, POSITION, split="validation")

    # --- controlli di allineamento ---
    gt_np = np.asarray(gt_list, dtype=np.int32)
    for k_ in test_outputs:
        assert test_outputs[k_].shape[0] == len(gt_np), f"Mismatch batch su {k_}"

    # ====== IMAGE-LEVEL: KNN su avgpool ======
    Xtest = torch.flatten(test_outputs['avgpool'], 1)   # (N_test, D)
    Xtrain = torch.flatten(train_outputs['avgpool'], 1) # (N_train, D)
    dist_matrix = calc_dist_matrix(Xtest, Xtrain)

    K_used = int(min(TOP_K, Xtrain.shape[0]))  # K dinamico per sicurezza
    topk_values, topk_indexes = torch.topk(dist_matrix, k=K_used, dim=1, largest=False)
    if K_used < TOP_K:
        print(f"[warn] KNN: TOP_K={TOP_K} ridotto a {K_used} (train size = {Xtrain.shape[0]})")

    scores = torch.mean(topk_values, dim=1).cpu().numpy()  # (N_test,)

    # selezione soglia per accuracy massima (usa solo gt image-level)
    fpr, tpr, thresholds = roc_curve(gt_np, scores)
    accs = []
    for th in thresholds:
        preds = (scores >= th).astype(np.int32)
        accs.append(accuracy_score(gt_np, preds))
    best_idx = int(np.argmax(accs))
    best_thr = thresholds[best_idx]
    preds    = (scores >= best_thr).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0, 1]).ravel()
    print(f"[image-level] threshold selezionata: {best_thr:.6f}")
    print(f"[image-level] immagini ANOMALE predette: {int(preds.sum())} su {len(preds)}")
    print(f"[image-level] accuracy: {accs[best_idx]*100:.2f}%")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")

    # --- mostra tutte le immagini validation con il loro anomaly score ---
    N = len(gt_np)
    per_page = 2         # quante immagini per pagina
    cols = 4

    print(f"[check] len(val_loader.dataset) = {len(val_loader.dataset)}")
    print(f"[check] len(scores)             = {len(scores)}")

    show_validation_grid_from_loader(val_loader.dataset, scores, preds, best_thr, per_page, cols)

    # -------------- pixel level anomaly -----------------------------
    score_map_list = []
    for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % CODICE_PEZZO):
        per_layer_maps = []
        for layer_name in ['layer1', 'layer2', 'layer3']:  # layer usati per localization
            # K vicini per questa immagine (dagli indici image-level)
            K = topk_indexes.shape[1]

            # estraggo feature map dei K vicini e della test
            topk_feat = train_outputs[layer_name][topk_indexes[t_idx]].to(device)   # (K,C,H,W)
            test_feat = test_outputs[layer_name][t_idx:t_idx + 1].to(device)        # (1,C,H,W)

            # L2-norm sul canale → distanza cosine-like più stabile
            topk_feat = l2norm(topk_feat, dim=1)
            test_feat = l2norm(test_feat, dim=1)

            K_, C, H, W = topk_feat.shape

            # Galleria: tutti i pixel di tutti i K vicini → (K*H*W, C)
            gallery = topk_feat.permute(0, 2, 3, 1).reshape(K_*H*W, C).contiguous()
            # Query: tutti i pixel della test → (H*W, C)
            query   = test_feat.permute(0, 2, 3, 1).reshape(H*W, C).contiguous()

            # cdist a blocchi (no perdita di vettori, no broadcasting ambiguo)
            B = 20000
            mins = []
            for s in range(0, gallery.shape[0], B):
                d = torch.cdist(gallery[s:s+B], query)     # (B, H*W)
                mins.append(d.min(dim=0).values)           # min su galleria per ciascun pixel test
            dist_min = torch.stack(mins, dim=0).min(dim=0).values  # (H*W,)

            # mappa (H,W) → upsample a 224
            score_map = dist_min.view(1, 1, H, W)
            score_map = F.interpolate(score_map, size=IMG_SIZE, mode='bilinear', align_corners=False)
            per_layer_maps.append(score_map.cpu())

        # media tra layer (texture + semantica)
        score_map = torch.mean(torch.cat(per_layer_maps, dim=0), dim=0)  # (1,1,224,224)
        score_map = score_map.squeeze().numpy()
        if GAUSSIAN_SIGMA > 0:
            score_map = gaussian_filter(score_map, sigma=GAUSSIAN_SIGMA)
        score_map_list.append(score_map)

    # --- NORMALIZZAZIONE score map GLOBALE (solo sulle good) ---
    good_maps = [sm for sm, y in zip(score_map_list, gt_np) if y == 0]
    if len(good_maps) == 0:
        good_maps = score_map_list  # fallback: tutte
    global_min = np.min([sm.min() for sm in good_maps])
    global_max = np.max([sm.max() for sm in good_maps])

    maps_norm = []
    for sm in score_map_list:
        sm_n = (sm - global_min) / (global_max - global_min + 1e-8)
        maps_norm.append(sm_n)

    # soglia percentile SOLO su pixel delle immagini good (normalizzate globalmente)
    pix_good = []
    for sm_n, y in zip(maps_norm, gt_np):
        if y == 0:
            pix_good.append(sm_n.ravel())
    if len(pix_good) == 0:
        pix_good = [sm_n.ravel() for sm_n in maps_norm]
    pix_good = np.concatenate(pix_good)
    thr_pix = float(np.percentile(pix_good, P_PIX))
    print(f"[pixel-level] soglia percentile {P_PIX}% (su pixel delle immagini good, normalizzazione globale) = {thr_pix:.4f}")

    # maschere binarie
    masks = [(sm_n >= thr_pix).astype(np.uint8) for sm_n in maps_norm]
    print(f"[pixel-level] immagini: {len(masks)} | threshold globale (percentile) = {thr_pix:.4f}")

    # Visualizza
    show_pixel_localization_grid(
        dataset=val_set,
        score_maps_norm=maps_norm,
        masks=masks,
        rows=2, cols=3,
        cmap_name="jet",
        suptitle=f"Pixel localization (percentile={P_PIX}%, thr={thr_pix:.3f}, global norm)"
    )

    # --- Soglia manuale su mappe normalizzate globalmente (opzionale) ---
    MANUAL_TRESHOLD = 0.65
    masks_manual = [(sm_n >= MANUAL_TRESHOLD).astype(np.uint8) for sm_n in maps_norm]
    print(f"[pixel-level] soglia usata: {MANUAL_TRESHOLD} | immagini: {len(masks_manual)}")
    show_pixel_localization_grid(
        dataset=val_set,
        score_maps_norm=maps_norm,
        masks=masks_manual,
        rows=2, cols=3,
        cmap_name="jet",
        suptitle=f"Pixel localization (thr={MANUAL_TRESHOLD}, global norm)"
    )

if __name__ == "__main__":
    main()
