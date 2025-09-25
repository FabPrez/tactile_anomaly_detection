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
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

# >>> NEW: per componenti connesse nelle GT
from scipy.ndimage import label as cc_label

# miei pacchetti
from data_loader import get_items, save_split_pickle, load_split_pickle
from view_utils import show_dataset_images, show_validation_grid_from_loader, show_heatmaps_from_loader
from ad_analysis import run_pixel_level_evaluation, print_pixel_report


# ----------------- CONFIG -----------------
METHOD = "SPADE"
CODICE_PEZZO = "PZ1"
POSITION    = "pos3"    # oppure "all"
TOP_K       = 7
IMG_SIZE    = 224       # input ResNet
SEED        = 42

VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = True
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

def l2norm(x, dim=1, eps=1e-6):
    """Normalizzazione L2 lungo la dimensione 'dim' (cosine-like distance)."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ---------- dataset ----------
class MyDataset(Dataset):
    def __init__(self,
                 images,                     # Sequence[PIL.Image.Image] (RGB)
                 label_type: str,            # "good" -> 0, "fault" -> 1
                 transform=None,             # es. transforms.CenterCrop(IMG_SIZE)
                 masks=None                  # opzionale: Sequence[mask] (np.ndarray HxW o PIL), oppure None
                 ):
        assert label_type in {"good", "fault"}, "label_type deve essere 'good' o 'fault'"
        self.label_value = 0 if label_type == "good" else 1
        self.transform = transform
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        # normalizza la lista maschere alla stessa lunghezza delle immagini
        if masks is None:
            masks = [None] * len(images)
        else:
            assert len(masks) == len(images), "masks deve avere stessa lunghezza di images"

        imgs_t, masks_t = [], []

        for img_pil, mk in zip(images, masks):
            # --- immagine ---
            assert isinstance(img_pil, Image.Image), "Le immagini devono essere PIL.Image"
            if transform is not None:
                img_pil = transform(img_pil)       # ancora PIL
            img_np = np.array(img_pil)             # (H,W,C) uint8
            img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (C,H,W) in [0,1]
            
            # img_t = (img_t - mean) / std # comment if you don't want to normalize with respect to typical resnet parameters
            
            H, W = img_t.shape[1], img_t.shape[2]

            # --- maschera ---
            if mk is None:                          # good o mask mancante → tutta zero
                mk_np = np.zeros((H, W), dtype=np.uint8)
            else:
                if isinstance(mk, Image.Image):
                    mk_pil = mk.convert("L") # dovrebbe convertire correttamente il nero in 0 e il resto in 1
                else:
                    # np.ndarray
                    mk_np0 = np.array(mk)           # può essere 0/1 o 0..255
                    mk_pil = Image.fromarray(mk_np0.astype(np.uint8), mode="L")
                # applica stesso transform geometrico (CenterCrop non interpola)
                if transform is not None:
                    mk_pil = transform(mk_pil)
                mk_np = np.array(mk_pil, dtype=np.uint8)
                # binarizza robustamente
                mk_np = (mk_np > 0).astype(np.uint8)
                # assicura stessa HxW dell'immagine
                if mk_np.shape != (H, W):
                    # fallback, dovrebbe raramente servire con stesso transform
                    mk_pil = Image.fromarray(mk_np, mode="L").resize((W, H), resample=Image.NEAREST)
                    mk_np = np.array(mk_pil, dtype=np.uint8)

            imgs_t.append(img_t)
            masks_t.append(torch.from_numpy(mk_np))    # (H,W) uint8 {0,1}

        self.data   = torch.stack(imgs_t)                                # (N, C, H, W)
        self.labels = torch.full((len(self.data),), self.label_value, dtype=torch.long)  # (N,)
        self.masks  = torch.stack(masks_t)                                # (N, H, W) uint8 {0,1}

        print('data shape ', self.data.shape)
        print('label shape', self.labels.shape)
        print('mask shape ', self.masks.shape)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.masks[idx]

    def __len__(self):
        return len(self.data)



def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("**device found:", device)

    # transform deterministica
    pre_processing = transforms.Compose([transforms.CenterCrop(IMG_SIZE)])

    # ---- carico immagini + (per fault) maschere ----
    good_imgs_pil = get_items(CODICE_PEZZO, "rgb", label="good", positions=POSITION, return_type="pil")
    
    fault_imgs_pil, fault_masks_np = get_items(
        CODICE_PEZZO, "rgb", label="fault", positions=POSITION,
        return_type="pil",
        with_masks=True, mask_return_type="numpy", mask_binarize=True,  # binarie 0/1
        mask_align="name" 
    )
    
    # ---- dataset ----
    good_ds  = MyDataset(good_imgs_pil,  label_type='good',  transform=pre_processing, masks=None)
    fault_ds = MyDataset(fault_imgs_pil, label_type='fault', transform=pre_processing, masks=fault_masks_np)
    
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
    
    if VIS_VALID_DATASET:
        show_dataset_images(val_set, batch_size=5, show_mask=True)
        
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
        train_outputs = load_split_pickle(CODICE_PEZZO, POSITION, split="train", method=METHOD)
        print("[cache] Train features caricate da pickle.")
    except FileNotFoundError:
        print("[cache] Nessun pickle train: estraggo feature...")
        for x, y, m in tqdm(train_loader, desc='| feature extraction | train | custom |'):
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                _ = model(x)
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.detach())
            outputs = []
        for k in train_outputs:
            train_outputs[k] = torch.cat(train_outputs[k], dim=0)
        save_split_pickle(train_outputs, CODICE_PEZZO, POSITION, split="train", method=METHOD)
    
    # ci aspettiamo una struttura così di train_outputs:
    # {
    #     'layer1': Tensor (N, C1, H1, W1),
    #     'layer2': Tensor (N, C2, H2, W2),
    #     'layer3': Tensor (N, C3, H3, W3),
    #     'avgpool': Tensor (N, 2048, 1, 1),
    # }
    # dove N è il numero di immagini. Mentre AVGPOOL è l'ultimo strato della resnet in questione, questo va ad utilizzare 2048 numeri per descrivere l'intera feature map, mediando sui canali

    # ====== VAL FEATURES (cache) ======
    try:
        pack         = load_split_pickle(CODICE_PEZZO, POSITION, split="validation", method=METHOD)
        test_outputs = pack['features']
        gt_list      = pack['labels']
        print("[cache] Validation features caricate da pickle.")
    except FileNotFoundError:
        print("[cache] Nessun pickle validation: estraggo feature...")
        gt_list = []
        for x, y, m in tqdm(val_loader, desc='| feature extraction | validation | custom |'):
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
        save_split_pickle(pack, CODICE_PEZZO, POSITION, split="validation", method=METHOD)

    # --- controlli di allineamento ---
    gt_np = np.asarray(gt_list, dtype=np.int32)
    for k in test_outputs:
        assert test_outputs[k].shape[0] == len(gt_np), f"Mismatch batch su {k}"

    # ====== IMAGE-LEVEL: KNN su avgpool ======
    dist_matrix = calc_dist_matrix(
        torch.flatten(test_outputs['avgpool'], 1),    # (N_test, D)
        torch.flatten(train_outputs['avgpool'], 1))   # (N_train, D)
    
    # vado a calcolare la distanza euclidea considerenado un immagine fatta da 2048 numeri
    topk_values, topk_indexes = torch.topk(dist_matrix, k=TOP_K, dim=1, largest=False)
    img_scores = torch.mean(topk_values, dim=1).cpu().numpy()  # (N_test,)
    
    fpr, tpr, thresholds = roc_curve(gt_np, img_scores)
    auc_img = roc_auc_score(gt_np, img_scores)
    
    print(f"[image-level] ROC-AUC ({CODICE_PEZZO}/{POSITION}): {auc_img:.3f}")
    
    # find the best treshold for the classifcation according to Youden's J
    J = tpr-fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])

    # Predizioni a soglia Youden
    preds = (img_scores >= best_thr).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(gt_np, preds, labels=[0, 1]).ravel()
    print(f"[image-level] soglia (Youden) = {best_thr:.6f}")
    print(f"[image-level] CM -> TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    print(f"[image-level] TPR:{fpr[best_idx]:.3f}  FPR:{tpr[best_idx]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(gt_list, img_scores):.3f}")
    ax[0].plot([0,1],[0,1],'k--',linewidth=1)
    ax[0].set_title("Image-level ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()

    plt.tight_layout(); plt.show()

    # --- SHOW IMAGES ---
    N = len(gt_np)
    per_page = 2         
    cols = 4
    
    print(f"[check] len(val_loader.dataset) = {len(val_loader.dataset)}")
    print(f"[check] len(scores)             = {len(img_scores)}")

    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True, overlay_alpha=0.45
        )

    
    # ---- PIXEL LEVEL FEATURES --------------------------
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

            # cdist a blocchi
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
        
    # ---- Valutazione & visualizzazione (riusabile dai tuoi altri metodi) ----
    results = run_pixel_level_evaluation(
        score_map_list=score_map_list,
        val_set=val_set,
        img_scores=img_scores,
        use_threshold="pro",   # "roc" | "pr" | "pro"
        fpr_limit=0.01,         # resta di default 0.3
        vis=True,
        vis_ds_or_loader=val_loader.dataset
    )

    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/{POSITION}")
        
if __name__ == "__main__":
    main()
