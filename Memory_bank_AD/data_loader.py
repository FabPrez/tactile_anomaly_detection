# data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import List, Iterable, Union, Literal, Optional
import pickle
import re

from PIL import Image
import numpy as np

# ================== PATH & COSTANTI ==================
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "Dataset"

# Estensioni supportate
EXTS = {
    "rgb": {".png"},
    "pointcloud": {".ply"},
}
EXTS_MASK = {".png"}  # maschere (binary png)

# ---------------- natural sort ----------------
_num_re = re.compile(r"(\d+)")
def _natsort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(s)]

def _iter_files(
    root: Optional[Path],
    exts: set[str],
    sort_mode: Literal["natural", "name", "mtime"] = "natural",
) -> List[Path]:
    if not root or not root.exists():
        return []
    exts_low = {e.lower() for e in exts}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_low]
    if sort_mode == "natural":
        files.sort(key=lambda p: _natsort_key(p.name))
    elif sort_mode == "name":
        files.sort(key=lambda p: p.name.lower())
    elif sort_mode == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: p.name.lower())
    return files

# ---------------- util di base ----------------
def list_positions(part: str) -> List[str]:
    base = DATASET_ROOT / part
    if not base.exists():
        raise FileNotFoundError(f"Part non trovato: {base}")
    return sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.lower().startswith("pos"))

def _normalize_positions(part: str, positions: Union[None, str, Iterable[str]]) -> List[str]:
    if positions is None or positions == "*" or positions == "all":
        return list_positions(part)
    if isinstance(positions, str):
        return [positions]
    return list(positions)

def _load_rgb_pil(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img

def _load_mask_pil(path: Path) -> Image.Image:
    img = Image.open(path).convert("L")
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img

def _norm_stem_for_match(stem: str) -> str:
    s = stem.lower()
    for suf in ("_mask", "-mask", ".mask", " mask"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

# ---------------- API di IO: immagini & maschere ----------------
def get_items(
    part: str,
    modality: str,
    label: str = "good",
    positions: Union[None, str, Iterable[str]] = None,
    return_type: Literal["path", "pil", "numpy"] = "path",
    *,
    with_masks: bool = False,
    mask_return_type: Literal["path", "pil", "numpy"] = "numpy",
    mask_binarize: bool = True,     # se True: numpy 0/1 (o PIL con 0/255)
    mask_align: Literal["order", "name"] = "order",  # mantenuto per compatibilità (qui usiamo l'indice)
) -> Union[
    List[Union[Path, Image.Image, np.ndarray]],
    tuple[List[Union[Path, Image.Image, np.ndarray]], List[Optional[Union[Path, Image.Image, np.ndarray]]]]
]:
    """
    Layout usato:
      - GOOD  -> immagini da: pos/rgb_320x240  (fallback pos/rgb)
      - FAULT -> immagini da: pos/fault/rgb_320x240 -> pos/fault/rgb -> pos/rgb_320x240 -> pos/rgb
                 maschere da: pos/fault/mask_320x240 -> pos/fault/mask
      - FAULT: abbinamento img–mask per indice (natural sort), fino a min(#img, #mask).
    """
    modality = modality.lower()
    label = label.lower()
    if modality not in EXTS:
        raise ValueError("modality deve essere 'rgb' o 'pointcloud'")
    if label not in {"good", "fault"}:
        raise ValueError("label deve essere 'good' oppure 'fault'")

    base = DATASET_ROOT / part
    pos_list = _normalize_positions(part, positions)

    results_paths: List[Path] = []
    mask_paths_all: List[Optional[Path]] = []

    for pos in pos_list:
        base_pos = base / pos

        # ----- scegli cartella immagini in base al label -----
        if modality == "rgb":
            if label == "fault":
                cand_imgs = [
                    base_pos / "fault" / "rgb_320x240",
                    base_pos / "fault" / "rgb",
                    base_pos / "rgb_320x240",
                    base_pos / "rgb",
                ]
            else:  # GOOD
                cand_imgs = [
                    base_pos / "rgb_320x240",
                    base_pos / "rgb",
                ]
        else:
            cand_imgs = [base_pos / "pointcloud_320x240", base_pos / "pointcloud"]

        img_dir = next((p for p in cand_imgs if p.exists()), None)
        imgs_here = _iter_files(img_dir, EXTS[modality], sort_mode="natural") if img_dir else []

        if not img_dir:
            print(f"[!] Cartella immagini non trovata per {part}/{pos}/{label}/{modality}. Provati: {cand_imgs}")
            continue
        else:
            total = len(list(img_dir.glob("*")))
            print(f"[{part}/{pos}/{label}/{modality}] Trovati {len(imgs_here)} di {total} file in {img_dir.name}")

        # ----- FAULT: abbina immagini e maschere per INDICE -----
        if (label == "fault") and (modality == "rgb"):
            cand_masks = [
                base_pos / "fault" / "mask_320x240",
                base_pos / "fault" / "mask",
            ]
            mask_dir = next((p for p in cand_masks if p.exists()), None)
            masks_here = _iter_files(mask_dir, EXTS_MASK, sort_mode="natural") if mask_dir else []

            n = min(len(imgs_here), len(masks_here))
            if n == 0:
                print(f"[WARN] {part}/{pos}: nessuna coppia img/mask (img={len(imgs_here)}, mask={len(masks_here)})")
            else:
                if len(imgs_here) != len(masks_here):
                    print(f"[INFO] {part}/{pos}: allineamento per indice -> usati {n} pair "
                          f"(img={len(imgs_here)}, mask={len(masks_here)})")
                results_paths.extend(imgs_here[:n])
                if with_masks:
                    mask_paths_all.extend(masks_here[:n])
            continue  # prossima posizione

        # ----- GOOD (e altri casi): prendi tutto dalla cartella scelta -----
        results_paths.extend(imgs_here)
        if with_masks:
            mask_paths_all.extend([None] * len(imgs_here))

    # ---- solo immagini ----
    if not with_masks:
        if return_type == "path":
            return results_paths

        if modality != "rgb":
            raise ValueError("return_type 'pil' o 'numpy' supportato solo per modality='rgb'.")

        out_imgs: List[Union[Image.Image, np.ndarray]] = []
        for p in results_paths:
            pil_img = _load_rgb_pil(p)
            if return_type == "pil":
                out_imgs.append(pil_img)
            elif return_type == "numpy":
                out_imgs.append(np.array(pil_img))
            else:
                raise ValueError("return_type non valido.")
        return out_imgs

    # ---- con maschere ----
    if return_type == "path":
        imgs_out: List[Union[Path, Image.Image, np.ndarray]] = results_paths
    else:
        if modality != "rgb":
            raise ValueError("return_type 'pil' o 'numpy' supportato solo per modality='rgb'.")
        imgs_out = []
        for p in results_paths:
            pil_img = _load_rgb_pil(p)
            if return_type == "pil":
                imgs_out.append(pil_img)
            elif return_type == "numpy":
                imgs_out.append(np.array(pil_img))
            else:
                raise ValueError("return_type non valido.")

    masks_out: List[Optional[Union[Path, Image.Image, np.ndarray]]] = []
    for mp in mask_paths_all:
        if mp is None:
            masks_out.append(None)
            continue
        if mask_return_type == "path":
            masks_out.append(mp)
        elif mask_return_type == "pil":
            m = _load_mask_pil(mp)
            if mask_binarize:
                m = m.point(lambda v: 255 if v > 0 else 0)
            masks_out.append(m)
        elif mask_return_type == "numpy":
            m = np.array(_load_mask_pil(mp))  # (H,W), 0..255
            if mask_binarize:
                m = (m > 0).astype(np.uint8)   # 0/1
            masks_out.append(m)
        else:
            raise ValueError("mask_return_type non valido.")

    assert len(imgs_out) == len(masks_out), "imgs e masks devono avere stessa lunghezza"
    return imgs_out, masks_out

# ---------------- pickle helpers ----------------
def _features_dir_for(part: str, position: str | None) -> Path:
    pos_norm = None if position is None else str(position).lower()
    tag = "all" if pos_norm in (None, "all", "*") else str(position)
    return DATASET_ROOT / part / "features" / tag

def _normalize_method(method: str) -> str:
    m = method.strip().lower().replace(" ", "-")
    m = re.sub(r"[^a-z0-9_-]", "", m)
    if not m:
        raise ValueError("method non può essere vuoto dopo la normalizzazione")
    return m

def save_split_pickle(obj, part: str, position: str | None, split: str, method: str) -> Path:
    split = split.lower()
    if split not in {"train", "validation"}:
        raise ValueError("split deve essere 'train' oppure 'validation'")
    method_norm = _normalize_method(method)
    dirpath = _features_dir_for(part, position)
    dirpath.mkdir(parents=True, exist_ok=True)
    filename = f"{method_norm}_{split}.pickle"
    p = dirpath / filename
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[pickle] Salvato oggetto in {p}")
    return p

def load_split_pickle(part: str, position: str | None, split: str, method: str):
    split = split.lower()
    if split not in {"train", "validation"}:
        raise ValueError("split deve essere 'train' oppure 'validation'")
    method_norm = _normalize_method(method)
    dirpath = _features_dir_for(part, position)
    p = dirpath / f"{method_norm}_{split}.pickle"
    if not p.exists():
        raise FileNotFoundError(f"Pickle non trovato: {p}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    print(f"[pickle] Caricato oggetto da {p}")
    return obj

# ============================================================
# ========= GENERIC AD DATA BUILDERS (SPADE, PaDiM, ...) =====
# ============================================================
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _positions_tag(pos_list):
    return "+".join(pos_list) if len(pos_list) > 0 else "all"

def _train_tag_with_fraction(pos_list, good_fraction: float | None):
    base = _positions_tag(pos_list)
    if good_fraction is None or good_fraction >= 0.999:
        return base
    pct = max(1, int(round(100 * float(good_fraction))))
    return f"{base}@p{pct}"

def _resolve_val_fault_positions(all_pos, train_pos_list, scope):
    if isinstance(scope, (list, tuple, set)):
        return [p for p in all_pos if p in scope]
    scope = str(scope).lower()
    if scope == "train_only":
        return train_pos_list
    if scope == "all":
        return all_pos
    raise ValueError("VAL_FAULT_SCOPE non riconosciuto")

def _resolve_val_good_positions(all_pos, train_pos_list, scope):
    if isinstance(scope, (list, tuple, set)):
        return [p for p in all_pos if p in scope]
    scope = str(scope).lower()
    if scope == "from_train":
        return train_pos_list
    if scope == "all_positions":
        return all_pos
    if scope == "none":
        return []
    raise ValueError("VAL_GOOD_SCOPE non riconosciuto")

def _get_T():
    try:
        from torchvision import transforms as _t1  # noqa: F401
        from torchvision.transforms import v2 as T
        return T
    except Exception:
        import torchvision.transforms as T
        return T

class MyDataset(Dataset):
    """
    Dataset semplice per immagini RGB (lista PIL) con maschere opzionali (lista np.ndarray o PIL).
    Normalizza in [0,1], applica la stessa transform geometrica a immagine e maschera.
    """
    def __init__(self, images, label_type: str, transform=None, masks=None):
        assert label_type in {"good", "fault"}, "label_type deve essere 'good' o 'fault'"
        self.label_value = 0 if label_type == "good" else 1
        self.transform = transform

        if masks is None:
            masks = [None] * len(images)
        else:
            assert len(masks) == len(images), "masks deve avere stessa lunghezza di images"

        imgs_t, masks_t = [], []
        for img_pil, mk in zip(images, masks):
            assert isinstance(img_pil, Image.Image), "Le immagini devono essere PIL.Image"
            img_pil_t = transform(img_pil) if transform is not None else img_pil
            img_np = np.array(img_pil_t)             # (H,W,C) uint8
            img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (C,H,W)

            H, W = img_t.shape[1], img_t.shape[2]

            if mk is None:
                mk_np = np.zeros((H, W), dtype=np.uint8)
            else:
                if isinstance(mk, Image.Image):
                    mk_pil = mk.convert("L")
                else:
                    mk_np0 = np.array(mk)
                    mk_pil = Image.fromarray(mk_np0.astype(np.uint8), mode="L")
                mk_pil = transform(mk_pil) if transform is not None else mk_pil
                mk_np = np.array(mk_pil, dtype=np.uint8)
                mk_np = (mk_np > 0).astype(np.uint8)
                if mk_np.shape != (H, W):
                    mk_pil = Image.fromarray(mk_np, mode="L").resize((W, H), resample=Image.NEAREST)
                    mk_np = np.array(mk_pil, dtype=np.uint8)

            imgs_t.append(img_t)
            masks_t.append(torch.from_numpy(mk_np))

        if len(imgs_t) == 0:
            raise ValueError("Nessuna immagine nel dataset costruito (lista vuota). "
                             "Controlla percorsi/cartelle e split.")

        self.data   = torch.stack(imgs_t)                                # (N, C, H, W)
        self.labels = torch.full((len(self.data),), self.label_value, dtype=torch.long)
        self.masks  = torch.stack(masks_t)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.masks[idx]

    def __len__(self):
        return len(self.data)

def _cap_warn(pos, want, avail):
    k = min(int(want), int(avail))
    if k < want:
        print(f"[warn] VAL_GOOD_PER_POS: richiesti {want} good in {pos}, ma disponibili {avail}. Uso {k}.")
    return k

def _k_map_for_positions(positions, val_good_per_pos):
    if val_good_per_pos is None:
        return {p: 0 for p in positions}
    if isinstance(val_good_per_pos, int):
        return {p: max(val_good_per_pos, 0) for p in positions}
    if isinstance(val_good_per_pos, dict):
        return {p: max(int(val_good_per_pos.get(p, 0)), 0) for p in positions}
    raise ValueError("val_good_per_pos deve essere int, dict o None")

# ---------------- builder principale ----------------
def build_ad_datasets(
    *,
    part: str,
    img_size: int,
    train_positions,            # str | list[str]
    val_fault_scope,            # "train_only" | "all" | list[str]
    val_good_scope,             # "from_train" | "all_positions" | "none" | list[str]
    val_good_per_pos: int | dict | None,   # quante GOOD per pos da spostare in validation
    good_fraction: float | None = 1.0,     # % delle GOOD rimanenti da usare per il TRAIN per posizione
    seed: int = 42,
    transform=None,
):
    """
    - Seleziona per posizione k=VAL_GOOD_PER_POS immagini GOOD per la validation.
    - Rimuove queste GOOD dal TRAIN, poi applica il sotto-campionamento per-posizione con good_fraction.
    - Ritorna (train_set, val_set, meta).
    """
    T = _get_T()
    if transform is None:
        try:
            transform = T.CenterCrop(img_size)
        except Exception:
            transform = T.Compose([T.CenterCrop(img_size)])

    all_pos = list_positions(part)
    train_pos_list = _ensure_list(train_positions)
    train_tag = _train_tag_with_fraction(train_pos_list, good_fraction)

    # pos candidate per VALIDATION good
    val_good_pos = _resolve_val_good_positions(all_pos, train_pos_list, val_good_scope)
    if str(val_good_scope).lower() == "none":
        k_map = {p: 0 for p in val_good_pos}
    else:
        k_map = _k_map_for_positions(val_good_pos, val_good_per_pos)

    # FAULT validation (img da fault/rgb_320x240*, mask da fault/mask_320x240*)
    val_fault_pos = _resolve_val_fault_positions(all_pos, train_pos_list, val_fault_scope)
    fault_val_pil, fault_val_masks = get_items(
        part, "rgb", label="fault", positions=val_fault_pos, return_type="pil",
        with_masks=True, mask_return_type="numpy", mask_binarize=True, mask_align="order"
    )

    rng = torch.Generator().manual_seed(seed)

    # manteniamo per posizione
    train_keep_by_pos: dict[str, list[Image.Image]] = {}
    val_sel_by_pos:   dict[str, list[Image.Image]] = {}
    per_pos_counts:   dict[str, dict] = {}

    # 1) posizioni del TRAIN: splitto togliendo k per la val
    for pos in train_pos_list:
        imgs_pos = get_items(part, "rgb", label="good", positions=[pos], return_type="pil")
        n = len(imgs_pos)
        k = int(k_map.get(pos, 0)) if pos in val_good_pos else 0
        if k > 0:
            k = _cap_warn(pos, k, n)
            idx = torch.randperm(n, generator=rng).tolist()
            sel  = [imgs_pos[i] for i in idx[:k]]
            keep = [imgs_pos[i] for i in idx[k:]]
        else:
            sel, keep = [], imgs_pos

        val_sel_by_pos[pos]    = sel
        train_keep_by_pos[pos] = keep
        per_pos_counts[pos] = {
            "good_total": n,
            "good_val":   len(sel),
            "good_train": len(keep),
        }

    # 2) posizioni extra per la validation (non in train)
    extra_val_pos = [p for p in val_good_pos if p not in train_pos_list]
    for pos in extra_val_pos:
        imgs_pos = get_items(part, "rgb", label="good", positions=[pos], return_type="pil")
        n = len(imgs_pos)
        k = int(k_map.get(pos, 0))
        k = _cap_warn(pos, k, n) if k > 0 else 0
        if k > 0:
            idx = torch.randperm(n, generator=rng).tolist()
            sel = [imgs_pos[i] for i in idx[:k]]
        else:
            sel = []
        val_sel_by_pos[pos] = sel
        per_pos_counts[pos] = {"good_total": n, "good_val": len(sel), "good_train": 0}

    # --- sotto-campionamento percentuale del TRAIN per posizione ---
    good_train_remaining: list[Image.Image] = []
    if good_fraction is None:
        good_fraction = 1.0
    good_fraction = float(good_fraction)

    for pos, keep in train_keep_by_pos.items():
        if good_fraction >= 0.999:
            picked = keep
        else:
            n_keep = len(keep)
            n_pick = max(0, int(round(n_keep * good_fraction)))
            if n_pick < n_keep:
                idx = torch.randperm(n_keep, generator=rng).tolist()
                picked = [keep[i] for i in idx[:n_pick]]
            else:
                picked = keep
        good_train_remaining.extend(picked)
        per_pos_counts[pos]["good_train_after_fraction"] = len(picked)

    # good validation globali
    good_val_selected: list[Image.Image] = []
    for pos, sel in val_sel_by_pos.items():
        good_val_selected.extend(sel)

    # --- dataset finali ---
    good_train_ds = MyDataset(good_train_remaining, label_type='good', transform=transform, masks=None)
    fault_val_ds  = MyDataset(fault_val_pil,  label_type='fault', transform=transform, masks=fault_val_masks)
    if len(good_val_selected) > 0:
        good_val_ds = MyDataset(good_val_selected, label_type='good', transform=transform, masks=None)
        val_set = ConcatDataset([good_val_ds, fault_val_ds])
    else:
        val_set = fault_val_ds

    meta = {
        "train_tag": train_tag,
        "good_fraction": good_fraction,
        "train_positions": train_pos_list,
        "val_fault_positions": val_fault_pos,
        "val_good_scope": val_good_scope,
        "val_good_positions": val_good_pos,
        "val_good_per_pos": k_map,
        "counts": {
            "train_good": len(good_train_ds),
            "val_good": len(good_val_selected),
            "val_fault": len(fault_val_ds),
            "val_total": (len(good_val_selected) + len(fault_val_ds)),
        },
        "per_pos_counts": per_pos_counts
    }
    return good_train_ds, val_set, meta

# ---------------- loaders ----------------
def make_loaders(train_set, val_set, batch_size=32, device: str | torch.device = "cpu"):
    pin = (str(device) == "cuda") or (isinstance(device, torch.device) and device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader
