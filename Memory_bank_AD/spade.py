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

# ---------- dataset ----------
class MyDataset(Dataset):
    def __init__(self, images, label_type: str, transform=None):
        """
        images: Sequence[PIL.Image.Image] (RGB)
        label_type: "good" -> 0, "fault" -> 1 (tutte uguali)
        transform: pre-process PIL→PIL (es. CenterCrop), **senza** ToTensor
        """
        assert label_type in {"good", "fault"}, "label_type deve essere 'good' o 'fault'"
        label_value = 0 if label_type == "good" else 1

        self.data = []
        for img_pil in images:
            assert isinstance(img_pil, Image.Image), "Le immagini devono essere PIL.Image"
            if transform is not None:
                img_pil = transform(img_pil)  # ancora PIL
            # PIL → Tensor (C,H,W) float32 in [0,1]
            img_t = torch.tensor(np.array(img_pil), dtype=torch.float32).permute(2, 0, 1) / 255.0
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

    # transform deterministica (ordine replicabile)
    pre_processing = transforms.Compose([transforms.CenterCrop(IMG_SIZE)])

    # carico immagini PIL
    good_imgs_pil  = get_items(CODICE_PEZZO, modality="rgb", label="good",  positions=POSITION, return_type="pil")
    fault_imgs_pil = get_items(CODICE_PEZZO, modality="rgb", label="fault", positions=POSITION, return_type="pil")

    # quick view
    # plt.imshow(good_imgs_pil[0]); plt.axis("off"); plt.show()

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
    
    # show_dataset_images(val_set, batch_size=5)

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
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.detach())
            outputs = []
        for k in train_outputs:
            train_outputs[k] = torch.cat(train_outputs[k], dim=0)
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
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.detach())
            outputs = []
        for k in test_outputs:
            test_outputs[k] = torch.cat(test_outputs[k], dim=0)
        pack = {'features': test_outputs, 'labels': np.array(gt_list, dtype=np.int64)}
        save_split_pickle(pack, CODICE_PEZZO, POSITION, split="validation")

    # --- controlli di allineamento ---
    gt_np = np.asarray(gt_list, dtype=np.int32)
    for k in test_outputs:
        assert test_outputs[k].shape[0] == len(gt_np), f"Mismatch batch su {k}"

    # ====== IMAGE-LEVEL: KNN su avgpool ======
    dist_matrix = calc_dist_matrix(
        torch.flatten(test_outputs['avgpool'], 1),    # (N_test, D)
        torch.flatten(train_outputs['avgpool'], 1))   # (N_train, D)

    topk_values, topk_indexes = torch.topk(dist_matrix, k=TOP_K, dim=1, largest=False)
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
        score_maps = []
        for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

            # construct a gallery of features at all pixel locations of the K nearest neighbors
            topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
            test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
            feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

            # calculate distance matrix
            dist_matrix_list = []
            for d_idx in range(feat_gallery.shape[0] // 100):
                dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                dist_matrix_list.append(dist_matrix)
            dist_matrix = torch.cat(dist_matrix_list, 0)

            # k nearest features from the gallery (k=1)
            score_map = torch.min(dist_matrix, dim=0)[0]
            score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                        mode='bilinear', align_corners=False)
            score_maps.append(score_map)
        # average distance between the features
        score_map = torch.mean(torch.cat(score_maps, 0), dim=0)
        # apply gaussian smoothing on the score map
        score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=GAUSSIAN_SIGMA)
        score_map_list.append(score_map)

    # --- NORMALIZZAZIONE score map GLOBALE (solo sulle good) ---
    # 1. Calcola min e max globali sulle score map delle sole immagini good
    good_maps = [sm for sm, y in zip(score_map_list, gt_np) if y == 0]
    if len(good_maps) == 0:
        good_maps = score_map_list  # fallback: tutte
    global_min = np.min([sm.min() for sm in good_maps])
    global_max = np.max([sm.max() for sm in good_maps])

    # 2. Normalizza tutte le score map usando questi valori globali
    maps_norm = []
    for sm in score_map_list:
        sm_n = (sm - global_min) / (global_max - global_min + 1e-8)
        maps_norm.append(sm_n)

    # 3. Calcola la soglia percentile SOLO sui pixel delle immagini good (normalizzate globalmente)
    pix_good = []
    for sm_n, y in zip(maps_norm, gt_np):
        if y == 0:
            pix_good.append(sm_n.ravel())
    if len(pix_good) == 0:
        pix_good = [sm_n.ravel() for sm_n in maps_norm]
    pix_good = np.concatenate(pix_good)
    thr_pix = float(np.percentile(pix_good, P_PIX))
    print(f"[pixel-level] soglia percentile {P_PIX}% (su pixel delle immagini good, normalizzazione globale) = {thr_pix:.4f}")

    # 4. Applica la soglia a tutte le score map normalizzate globalmente
    masks = []
    for sm_n in maps_norm:
        masks.append((sm_n >= thr_pix).astype(np.uint8))

    print(f"[pixel-level] immagini: {len(masks)} | threshold globale (percentile) = {thr_pix:.4f}")

    # 5. Visualizza
    show_pixel_localization_grid(
        dataset=val_set,
        score_maps_norm=maps_norm,
        masks=masks,
        rows=2, cols=3,
        cmap_name="jet",
        suptitle=f"Pixel localization (percentile={P_PIX}%, thr={thr_pix:.3f}, global norm)"
    )

    # --- Soglia manuale su mappe normalizzate globalmente ---
    MANUAL_TRESHOLD = 0.65
    masks_manual = []
    for sm_n in maps_norm:
        masks_manual.append((sm_n >= MANUAL_TRESHOLD).astype(np.uint8))
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
