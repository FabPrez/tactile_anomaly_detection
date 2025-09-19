import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch
import torch.nn.functional as F

def chw_to_hwc_uint8(t):
    # change the order of the channels passing from a tensor form to a MatplotLib or OpenCv format
    """(C,H,W) [0,1] -> (H,W,C) uint8"""
    """Tensor (C,H,W) in [0,1] -> np.uint8 (H,W,C)."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    img = np.transpose(t, (1, 2, 0))          # HWC
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def show_dataset_images(ds_or_loader, batch_size=8):
    """
    Mostra le immagini a pagine da `batch_size`.
    Puoi passare sia un Dataset che un DataLoader.
    """
    # Se è un Dataset, wrappalo in un DataLoader per ottenere batch (B,C,H,W)
    if isinstance(ds_or_loader, DataLoader):
        loader = ds_or_loader
    else:
        loader = DataLoader(ds_or_loader, batch_size=batch_size, shuffle=False)

    for batch_imgs, batch_labels in loader:
        # batch_imgs: (B,C,H,W), batch_labels: (B,)
        b = batch_imgs.size(0)
        fig, axes = plt.subplots(1, b, figsize=(4*b, 4))
        if b == 1:
            axes = [axes]  # uniforma l'indice

        for i in range(b):
            img = chw_to_hwc_uint8(batch_imgs[i])
            lab = int(batch_labels[i].item())
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"label: {lab}", color=('green' if lab == 0 else 'red'))

        plt.tight_layout()
        plt.show()

 

def show_validation_grid_from_loader(dataset,
                           scores: np.ndarray,
                           preds: np.ndarray | None = None,
                           threshold: float | None = None,
                           rows: int = 3,
                           cols: int = 4,
                           title_fmt: str = "idx {i} | score {s:.3f} | gt {g}"):
    """
    Mostra il validation set a pagine (rows x cols).
    - dataset: oggetto Dataset che restituisce (img_tensor(C,H,W in [0,1]), label 0/1)
    - scores: array (N,) con uno score per immagine, stesso ordine del dataset
    - preds:  array (N,) opzionale con predizioni 0/1; se None e threshold è dato, verrà calcolato
    - threshold: soglia opzionale per derivare preds da scores
    - rows, cols: layout della griglia per pagina
    - title_fmt: formato titolo base (senza 'pred', che viene aggiunto se disponibile)
    """
    N = len(dataset)
    scores = np.asarray(scores).reshape(-1)
    print("N invece è", N)
    print("len scores", len(scores))
    assert scores.shape[0] == N, "scores deve avere lunghezza = len(dataset)"

    if preds is None and threshold is not None:
        preds = (scores >= float(threshold)).astype(int)
    if preds is not None:
        preds = np.asarray(preds).reshape(-1)
        assert preds.shape[0] == N, "preds deve avere lunghezza = len(dataset)"

    per_page = rows * cols
    loader = DataLoader(dataset, batch_size=per_page, shuffle=False)

    start_idx = 0
    for batch_imgs, batch_labels in loader:
        b = batch_imgs.size(0)
        fig = plt.figure(figsize=(4*cols, 3*rows))

        for k in range(b):
            i = start_idx + k
            img = chw_to_hwc_uint8(batch_imgs[k])
            g = int(batch_labels[k].item())
            s = float(scores[i])

            if preds is not None:
                p = int(preds[i])
                full_title = f"{title_fmt} | pred {p}".format(i=i, s=s, g=g)
                color = ('red' if p == 1 else 'green')
            else:
                full_title = title_fmt.format(i=i, s=s, g=g)
                color = 'black'

            r = k // cols
            c = k %  cols
            ax = plt.subplot(rows, cols, k+1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(full_title, color=color, fontsize=9)

        plt.tight_layout()
        plt.show()
        start_idx += b


def normalize_map(m: np.ndarray, eps: float = 1e-8):
    mmin, mmax = float(m.min()), float(m.max())
    return (m - mmin) / (mmax - mmin + eps)

def resize_map_to_image(score_map: np.ndarray, img_chw: torch.Tensor):
    """Ridimensiona la score map alla risoluzione dell'immagine (C,H,W)."""
    H, W = int(img_chw.shape[1]), int(img_chw.shape[2])
    sm = torch.from_numpy(score_map).float()[None, None, ...]  # (1,1,h,w)
    sm = F.interpolate(sm, size=(H, W), mode='bilinear', align_corners=False)
    return sm.squeeze().cpu().numpy()  # (H,W)


def make_pixel_masks_manual(score_map_list,
                            dataset,
                            threshold: float,
                            normalize_each: bool = True):
    """
    Crea maschere binarie dato un valore di soglia *manuale*.
    - score_map_list: lista di np.ndarray 2D (una per immagine)
    - dataset: Dataset del validation (stesso ordine usato per estrarre le feature)
    - threshold: soglia da applicare alle score map (se normalize_each=True, mettila in [0,1])
    - normalize_each: normalizza ogni score map in [0,1] prima di sogliare

    Ritorna:
      masks         : lista di array (H,W) uint8 {0,1}
      score_maps_n  : lista di array (H,W) normalizzate [0,1] (comode per visualizzazione)
    """
    masks = []
    score_maps_norm = []

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    i_global = 0
    for X, _ in loader:  # X: (B,3,H,W) in [0,1]
        for b in range(X.size(0)):
            img_chw = X[b]                        # (3,H,W)
            sm = score_map_list[i_global]         # (h,w) (es. 224x224)
            sm = resize_map_to_image(sm, img_chw) # -> (H,W)
            if normalize_each:
                sm = normalize_map(sm)
            score_maps_norm.append(sm)

            mask = (sm >= threshold).astype(np.uint8)
            masks.append(mask)
            i_global += 1

    assert i_global == len(score_map_list), "Mismatch score_map_list vs dataset."
    return masks, score_maps_norm


def show_pixel_localization_grid(dataset,
                                 score_maps_norm: list[np.ndarray],
                                 masks: list[np.ndarray],
                                 rows: int = 2,
                                 cols: int = 3,
                                 cmap_name: str = "jet",
                                 suptitle: str | None = None,
                                 alpha_heat: float = 0.45):
    """
    Per ogni immagine mostra 4 pannelli SEPARATI:
      [0] Originale
      [1] Originale + heat (overlay con alpha)
      [2] Mask binaria
      [3] Img * mask

    Layout: rows x (cols*4)
    """
    N = len(masks)
    assert len(score_maps_norm) == N, "len(masks) e len(score_maps_norm) devono coincidere"

    per_page = rows * cols
    loader = DataLoader(dataset, batch_size=per_page, shuffle=False, num_workers=0)

    start = 0
    for X, Y in loader:
        b = X.size(0)
        total_cols = cols * 4
        fig, axes = plt.subplots(rows, total_cols, figsize=(3.2 * total_cols, 3.2 * rows))
        if suptitle:
            fig.suptitle(suptitle, fontsize=12)

        if rows == 1:
            axes = np.expand_dims(axes, 0)
        if total_cols == 1:
            axes = np.expand_dims(axes, 1)

        for k in range(b):
            i = start + k
            r = k // cols
            c0 = (k % cols) * 4

            img_hwc = chw_to_hwc_uint8(X[k])
            heat    = score_maps_norm[i]            # (H,W) in [0,1]
            msk     = masks[i].astype(np.uint8)
            lab     = int(Y[k].item())

            ax0 = axes[r, c0 + 0]
            ax0.imshow(img_hwc); ax0.axis('off')
            ax0.set_title(f"idx {i} | label {lab}")

            ax1 = axes[r, c0 + 1]
            ax1.imshow(img_hwc)
            ax1.imshow(heat, cmap=cmap_name, alpha=alpha_heat)  # <— usa alpha_heat
            ax1.axis('off'); ax1.set_title("heat overlay")

            ax2 = axes[r, c0 + 2]
            ax2.imshow(msk, cmap='gray'); ax2.axis('off'); ax2.set_title("mask")

            img_masked = img_hwc.copy()
            img_masked[msk == 0] = 0
            ax3 = axes[r, c0 + 3]
            ax3.imshow(img_masked); ax3.axis('off'); ax3.set_title("img*mask")

        plt.tight_layout()
        plt.show()
        start += b