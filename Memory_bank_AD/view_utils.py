import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch
import torch.nn.functional as F

def chw_to_hwc_uint8(t):
    """(C,H,W) [0,1] -> (H,W,C) uint8"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    img = np.transpose(t, (1, 2, 0))
    return (img * 255.0).clip(0, 255).astype(np.uint8)

def mask_to_uint8(m):
    """(H,W) o (1,H,W) -> (H,W) uint8 {0,1}."""
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    if m.ndim == 3 and m.shape[0] == 1:  # (1,H,W) -> (H,W)
        m = m[0]
    # binarizza in modo robusto
    if m.max() > 1.0:
        m = (m > 127).astype(np.uint8)
    else:
        m = (m > 0.5).astype(np.uint8)
    return m

def show_dataset_images(
    ds_or_loader,
    batch_size: int = 8,
    show_mask: bool = False,
    overlay: bool = False,
    overlay_alpha: float = 0.45,
    overlay_cmap: str = "jet",
    title_with_label: bool = True,
):
    """
    Mostra il dataset a pagine.
    - Se show_mask=False: 1 pannello per esempio (solo immagine).
    - Se show_mask=True:  2 pannelli per esempio [immagine(+overlay opzionale) | maschera].

    ds_or_loader: Dataset o DataLoader.
      Ogni item può essere (img, label) o (img, label, mask).
      img: Tensor (C,H,W) in [0,1]
      mask: Tensor/ndarray (H,W) o (1,H,W) con {0,1}/{0,255}

    overlay è considerato solo se show_mask=True.
    """
    # Wrappa in DataLoader se serve
    if isinstance(ds_or_loader, DataLoader):
        loader = ds_or_loader
    else:
        loader = DataLoader(ds_or_loader, batch_size=batch_size, shuffle=False)

    for batch in loader:
        # batch può essere (X,Y) o (X,Y,M)
        if len(batch) == 2:
            X, Y = batch
            M = None
        else:
            X, Y, M = batch

        B = X.size(0)
        cols = 2 if show_mask else 1
        rows = B

        fig = plt.figure(figsize=(5 * cols, 3.5 * rows))

        for i in range(B):
            img = chw_to_hwc_uint8(X[i])

            # col 1: immagine (+overlay solo se show_mask=True)
            ax1 = plt.subplot(rows, cols, i * cols + 1)
            ax1.imshow(img)
            if show_mask:
                # usa maschera fornita o, se assente, una maschera nera
                if M is None:
                    # genera mask nera con stessa H,W
                    H, W = img.shape[:2]
                    msk_vis = np.zeros((H, W), dtype=np.uint8)
                else:
                    msk_vis = mask_to_uint8(M[i])
                if overlay:
                    ax1.imshow(msk_vis, cmap=overlay_cmap, alpha=overlay_alpha)
            ax1.axis('off')
            if title_with_label:
                lab = int(Y[i].item())
                ax1.set_title(
                    f"img | label: {lab}",
                    color=('green' if lab == 0 else 'red')
                )

            # col 2: maschera (solo se show_mask=True)
            if show_mask:
                ax2 = plt.subplot(rows, cols, i * cols + 2)
                ax2.imshow(msk_vis, cmap='gray')
                ax2.axis('off')
                ax2.set_title("mask")

        plt.tight_layout()
        plt.show()

# def normalize_map(m: np.ndarray, eps: float = 1e-8):
#     mmin, mmax = float(m.min()), float(m.max())
#     return (m - mmin) / (mmax - mmin + eps)

# def resize_map_to_image(score_map: np.ndarray, img_chw: torch.Tensor):
#     """Ridimensiona la score map alla risoluzione dell'immagine (C,H,W)."""
#     H, W = int(img_chw.shape[1]), int(img_chw.shape[2])
#     sm = torch.from_numpy(score_map).float()[None, None, ...]  # (1,1,h,w)
#     sm = F.interpolate(sm, size=(H, W), mode='bilinear', align_corners=False)
#     return sm.squeeze().cpu().numpy()  # (H,W)


# def make_pixel_masks_manual(score_map_list,
#                             dataset,
#                             threshold: float,
#                             normalize_each: bool = True):
#     """
#     Crea maschere binarie dato un valore di soglia *manuale*.
#     - score_map_list: lista di np.ndarray 2D (una per immagine)
#     - dataset: Dataset del validation (stesso ordine usato per estrarre le feature)
#     - threshold: soglia da applicare alle score map (se normalize_each=True, mettila in [0,1])
#     - normalize_each: normalizza ogni score map in [0,1] prima di sogliare

#     Ritorna:
#       masks         : lista di array (H,W) uint8 {0,1}
#       score_maps_n  : lista di array (H,W) normalizzate [0,1] (comode per visualizzazione)
#     """
#     masks = []
#     score_maps_norm = []

#     loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

#     i_global = 0
#     for X, _ in loader:  # X: (B,3,H,W) in [0,1]
#         for b in range(X.size(0)):
#             img_chw = X[b]                        # (3,H,W)
#             sm = score_map_list[i_global]         # (h,w) (es. 224x224)
#             sm = resize_map_to_image(sm, img_chw) # -> (H,W)
#             if normalize_each:
#                 sm = normalize_map(sm)
#             score_maps_norm.append(sm)

#             mask = (sm >= threshold).astype(np.uint8)
#             masks.append(mask)
#             i_global += 1

#     assert i_global == len(score_map_list), "Mismatch score_map_list vs dataset."
#     return masks, score_maps_norm


# def show_pixel_localization_grid(dataset,
#                                  score_maps_norm: list[np.ndarray],
#                                  masks: list[np.ndarray],
#                                  rows: int = 2,
#                                  cols: int = 3,
#                                  cmap_name: str = "jet",
#                                  suptitle: str | None = None,
#                                  alpha_heat: float = 0.45):
#     """
#     Per ogni immagine mostra 4 pannelli SEPARATI:
#       [0] Originale
#       [1] Originale + heat (overlay con alpha)
#       [2] Mask binaria
#       [3] Img * mask

#     Layout: rows x (cols*4)
#     """
#     N = len(masks)
#     assert len(score_maps_norm) == N, "len(masks) e len(score_maps_norm) devono coincidere"

#     per_page = rows * cols
#     loader = DataLoader(dataset, batch_size=per_page, shuffle=False, num_workers=0)

#     start = 0
#     for X, Y in loader:
#         b = X.size(0)
#         total_cols = cols * 4
#         fig, axes = plt.subplots(rows, total_cols, figsize=(3.2 * total_cols, 3.2 * rows))
#         if suptitle:
#             fig.suptitle(suptitle, fontsize=12)

#         if rows == 1:
#             axes = np.expand_dims(axes, 0)
#         if total_cols == 1:
#             axes = np.expand_dims(axes, 1)

#         for k in range(b):
#             i = start + k
#             r = k // cols
#             c0 = (k % cols) * 4

#             img_hwc = chw_to_hwc_uint8(X[k])
#             heat    = score_maps_norm[i]            # (H,W) in [0,1]
#             msk     = masks[i].astype(np.uint8)
#             lab     = int(Y[k].item())

#             ax0 = axes[r, c0 + 0]
#             ax0.imshow(img_hwc); ax0.axis('off')
#             ax0.set_title(f"idx {i} | label {lab}")

#             ax1 = axes[r, c0 + 1]
#             ax1.imshow(img_hwc)
#             ax1.imshow(heat, cmap=cmap_name, alpha=alpha_heat)  # <— usa alpha_heat
#             ax1.axis('off'); ax1.set_title("heat overlay")

#             ax2 = axes[r, c0 + 2]
#             ax2.imshow(msk, cmap='gray'); ax2.axis('off'); ax2.set_title("mask")

#             img_masked = img_hwc.copy()
#             img_masked[msk == 0] = 0
#             ax3 = axes[r, c0 + 3]
#             ax3.imshow(img_masked); ax3.axis('off'); ax3.set_title("img*mask")

#         plt.tight_layout()
#         plt.show()
#         start += b
def show_validation_grid_from_loader(
    ds_or_loader,
    scores: np.ndarray,
    preds: np.ndarray | None = None,
    threshold: float | None = None,
    per_page: int = 8,
    *,
    show_mask: bool = False,
    show_mask_product: bool = True,
    overlay: bool = False,
    overlay_alpha: float = 0.45,
    overlay_cmap: str = "jet",
    samples_per_row: int = 4,            # <-- quante IMMAGINI per riga
    title_fmt: str = "idx {i} | s {s:.3f} | gt {g} | pred {p}",
):
    """
    Visualizza il validation set a pagine.

    - `per_page`: quante immagini per pagina (indipendente dal batch_size del loader esterno)
    - `samples_per_row`: quante immagini per riga (i blocchi 'img | mask | img*mask' vengono ripetuti in orizzontale)
    """
    # --- usa sempre il dataset sottostante e ricrea un loader con per_page ---
    if isinstance(ds_or_loader, DataLoader):
        dataset = ds_or_loader.dataset
    else:
        dataset = ds_or_loader
    loader = DataLoader(dataset, batch_size=per_page, shuffle=False)

    N = len(dataset)
    scores = np.asarray(scores).reshape(-1)
    assert scores.shape[0] == N, "scores deve avere lunghezza = numero di esempi"

    if preds is None and threshold is not None:
        preds = (scores >= float(threshold)).astype(int)
    if preds is not None:
        preds = np.asarray(preds).reshape(-1)
        assert preds.shape[0] == N, "preds deve avere lunghezza = numero di esempi"

    # colonne per BLOCCO (1=solo img; +1 mask; +1 img*mask)
    cols_per_block = 1 + (1 if show_mask else 0) + (1 if (show_mask and show_mask_product) else 0)
    assert cols_per_block >= 1

    start_idx = 0
    for batch in loader:
        if len(batch) == 2:
            X, Y = batch
            M = None
        else:
            X, Y, M = batch

        b = X.size(0)
        # quante righe di blocchi mi servono, con samples_per_row blocchi per riga?
        n_block_rows = int(np.ceil(b / samples_per_row))
        total_cols   = samples_per_row * cols_per_block

        fig = plt.figure(figsize=(3.8 * total_cols, 3.2 * n_block_rows))

        for k in range(b):
            i_global = start_idx + k
            img = chw_to_hwc_uint8(X[k])
            g   = int(Y[k].item())
            s   = float(scores[i_global])
            p   = int(preds[i_global]) if preds is not None else None

            # posizione del BLOCCO nella griglia
            br = k // samples_per_row                 # riga di blocchi
            bc = k %  samples_per_row                 # colonna di blocchi
            base_col = bc * cols_per_block            # colonna di partenza di questo blocco

            # --- colonna 1 del blocco: immagine (+ overlay se richiesto) ---
            ax1 = plt.subplot(n_block_rows, total_cols, br * total_cols + base_col + 1)
            ax1.imshow(img)

            msk_vis = None
            if show_mask:
                if M is None:
                    H, W = img.shape[:2]
                    msk_vis = np.zeros((H, W), dtype=np.uint8)
                else:
                    msk_vis = mask_to_uint8(M[k])
                    if msk_vis.shape[:2] != img.shape[:2]:
                        m_pil = Image.fromarray((msk_vis * 255).astype(np.uint8), mode="L")
                        m_pil = m_pil.resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
                        msk_vis = (np.array(m_pil) > 127).astype(np.uint8)
                if overlay:
                    ax1.imshow(msk_vis, cmap=overlay_cmap, alpha=overlay_alpha)

            ax1.axis('off')
            if p is None:
                ttl = f"idx {i_global} | s {s:.3f} | gt {g}"
                color = 'black'
            else:
                ttl = title_fmt.format(i=i_global, s=s, g=g, p=p)
                color = ('red' if p == 1 else 'green')
            ax1.set_title(ttl, color=color, fontsize=9)

            # --- colonna 2 del blocco: mask ---
            next_col = 2
            if show_mask:
                ax2 = plt.subplot(n_block_rows, total_cols, br * total_cols + base_col + 2)
                ax2.imshow(msk_vis, cmap='gray')
                ax2.axis('off')
                ax2.set_title("mask")
            else:
                next_col = 1

            # --- colonna 3 del blocco: img*mask ---
            if show_mask and show_mask_product:
                img_masked = img.copy()
                img_masked[msk_vis == 0] = 0
                ax3 = plt.subplot(n_block_rows, total_cols, br * total_cols + base_col + 3)
                ax3.imshow(img_masked)
                ax3.axis('off')
                ax3.set_title("img*mask")

        plt.tight_layout()
        plt.show()
        start_idx += b
