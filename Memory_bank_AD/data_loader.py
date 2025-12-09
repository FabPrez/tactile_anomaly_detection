# data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import List, Iterable, Union, Literal, Optional, Tuple
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
    mask_binarize: bool = True,
    mask_align: Literal["order", "name"] = "order",
    rgb_policy: Literal["prefer_320", "prefer_fullres", "fullres_only"] = "prefer_320",
):
    """
    Layout usato:
      - GOOD  -> immagini da: (policy) tra rgb_320x240 e rgb
      - FAULT -> immagini da: (policy) tra fault/rgb_320x240 e fault/rgb (fallback anche a good/rgb)
                 maschere da: (policy) tra fault/mask_320x240 e fault/mask
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

        # ----- scegli cartella immagini in base al label & policy -----
        if modality == "rgb":
            if label == "fault":
                if rgb_policy == "prefer_320":
                    cand_imgs = [
                        base_pos / "fault" / "rgb_320x240",
                        base_pos / "fault" / "rgb",
                        base_pos / "rgb_320x240",
                        base_pos / "rgb",
                    ]
                elif rgb_policy == "prefer_fullres":
                    cand_imgs = [
                        base_pos / "fault" / "rgb",
                        base_pos / "fault" / "rgb_320x240",
                        base_pos / "rgb",
                        base_pos / "rgb_320x240",
                    ]
                elif rgb_policy == "fullres_only":
                    cand_imgs = [base_pos / "fault" / "rgb"]
                else:
                    raise ValueError(f"rgb_policy non valido: {rgb_policy}")
            else:  # GOOD
                if rgb_policy == "prefer_320":
                    cand_imgs = [
                        base_pos / "rgb_320x240",
                        base_pos / "rgb",
                    ]
                elif rgb_policy == "prefer_fullres":
                    cand_imgs = [
                        base_pos / "rgb",
                        base_pos / "rgb_320x240",
                    ]
                elif rgb_policy == "fullres_only":
                    cand_imgs = [base_pos / "rgb"]
                else:
                    raise ValueError(f"rgb_policy non valido: {rgb_policy}")
        else:
            if rgb_policy == "prefer_320":
                cand_imgs = [base_pos / "pointcloud_320x240", base_pos / "pointcloud"]
            elif rgb_policy in ("prefer_fullres", "fullres_only"):
                cand_imgs = [base_pos / "pointcloud", base_pos / "pointcloud_320x240"]
            else:
                cand_imgs = [base_pos / "pointcloud_320x240", base_pos / "pointcloud"]

        img_dir = next((p for p in cand_imgs if p.exists()), None)
        imgs_here = _iter_files(img_dir, EXTS[modality], sort_mode="natural") if img_dir else []

        if not img_dir:
            print(f"[!] Cartella immagini non trovata per {part}/{pos}/{label}/{modality}. Provati: {cand_imgs}")
            continue
        else:
            total = len(list(img_dir.glob("*")))
            # print(f"[{part}/{pos}/{label}/{modality}] Trovati {len(imgs_here)} di {total} file in {img_dir.name}")

        # ----- FAULT: abbina immagini e maschere per INDICE -----
        if (label == "fault") and (modality == "rgb"):
            if rgb_policy == "prefer_320":
                cand_masks = [
                    base_pos / "fault" / "mask_320x240",
                    base_pos / "fault" / "mask",
                ]
            elif rgb_policy == "prefer_fullres":
                cand_masks = [
                    base_pos / "fault" / "mask",
                    base_pos / "fault" / "mask_320x240",
                ]
            elif rgb_policy == "fullres_only":
                cand_masks = [base_pos / "fault" / "mask"]
            else:
                raise ValueError(f"rgb_policy non valido: {rgb_policy}")

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
            m = np.array(_load_mask_pil(mp))
            if mask_binarize:
                m = (m > 0).astype(np.uint8)
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

# --- NEW: frazione per-posizione ---
def _get_frac_for_pos(good_fraction, pos: str) -> float:
    """
    Restituisce la frazione da usare per una certa posizione.

    - se good_fraction è float (o int): stessa frazione per tutte le posizioni
    - se è dict: usa good_fraction[pos] se presente, altrimenti 1.0
    - se è None: default 1.0
    """
    if isinstance(good_fraction, dict):
        frac = float(good_fraction.get(pos, 1.0))
    elif good_fraction is None:
        frac = 1.0
    else:
        frac = float(good_fraction)
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    return frac

def _train_tag_with_fraction(pos_list, good_fraction) -> str:
    """
    Crea un tag testuale che codifica le posizioni di train e (eventualmente)
    le frazioni per-posizione, usato per nominare i pickle.

    Esempi:
      - good_fraction = 0.2, pos_list=["pos1","pos2"]
          -> "pos1+pos2@p20"
      - good_fraction = {"pos1":0.2, "pos2":0.05}
          -> "pos1@p20+pos2@p05"
    """
    base = _positions_tag(pos_list)
    if isinstance(good_fraction, dict):
        parts = []
        for p in pos_list:
            frac = _get_frac_for_pos(good_fraction, p)
            if frac >= 0.999:
                parts.append(p)
            else:
                pct = max(1, int(round(100 * frac)))
                parts.append(f"{p}@p{pct}")
        return "+".join(parts) if parts else base
    else:
        if good_fraction is None or float(good_fraction) >= 0.999:
            return base
        frac = float(good_fraction)
        pct = max(1, int(round(100 * frac)))
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
    Dataset semplice per immagini RGB (lista PIL) con maschere opzionali.
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
            img_np = np.array(img_pil_t)
            img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

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
            raise ValueError("Nessuna immagine nel dataset costruito (lista vuota).")

        self.data   = torch.stack(imgs_t)
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
    """
    Restituisce una mappa pos -> valore grezzo (float) che può essere:
      - >= 1.0 : numero assoluto di immagini da spostare in validation
      - 0 < v < 1.0 : frazione del numero di GOOD di quella posizione
      - 0.0 : nessuna immagine per quella posizione
    """
    if val_good_per_pos is None:
        return {p: 0.0 for p in positions}
    if isinstance(val_good_per_pos, (int, float)):
        v = float(val_good_per_pos)
        return {p: max(v, 0.0) for p in positions}
    if isinstance(val_good_per_pos, dict):
        out = {}
        for p in positions:
            v = float(val_good_per_pos.get(p, 0.0))
            out[p] = max(v, 0.0)
        return out
    raise ValueError("val_good_per_pos deve essere int, float, dict o None")

# ---------------- builder principale ----------------
def build_ad_datasets(
    *,
    part: str,
    img_size: Optional[Union[int, Tuple[int, int]]] = None,
    train_positions,
    val_fault_scope,
    val_good_scope,
    val_good_per_pos: int | float | dict | None,
    good_fraction: float | dict | None = 1.0,
    seed: int = 42,
    train_seed: int | None = None,
    transform=None,
    rgb_policy: Literal["prefer_320", "prefer_fullres", "fullres_only"] = "prefer_320",
    debug_print_val_paths: bool = False,   # <<< NEW
):
    """
    Costruisce:
      - train_set: solo GOOD (dalle posizioni train_positions),
                   eventualmente con frazione per posizione good_fraction.
      - val_set:   GOOD (da varie posizioni) + FAULT (dalle posizioni val_fault_scope).

    Il numero di GOOD in validation è controllato da val_good_per_pos.
    Il seed controlla la scelta casuale dei good spostati in validation (=> "test").
    Il train_seed controlla la scelta casuale del sottoinsieme di GOOD usato per il TRAIN.
    """
    T = _get_T()

    # Trasformazioni/resize
    if transform is None:
        if img_size is None:
            transform = None
        else:
            if isinstance(img_size, int):
                size_tuple = (img_size, img_size)
            elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
                size_tuple = (int(img_size[0]), int(img_size[1]))
            else:
                raise ValueError(f"img_size deve essere int oppure (H, W); trovato: {img_size!r}")
            transform = T.CenterCrop(size_tuple)

    all_pos = list_positions(part)
    train_pos_list = _ensure_list(train_positions)
    train_tag = _train_tag_with_fraction(train_pos_list, good_fraction)

    # pos candidate per VALIDATION good
    val_good_pos = _resolve_val_good_positions(all_pos, train_pos_list, val_good_scope)
    if str(val_good_scope).lower() == "none":
        k_map_raw = {p: 0.0 for p in val_good_pos}
    else:
        k_map_raw = _k_map_for_positions(val_good_pos, val_good_per_pos)

    # FAULT validation
    val_fault_pos = _resolve_val_fault_positions(all_pos, train_pos_list, val_fault_scope)
    fault_val_pil, fault_val_masks = get_items(
        part, "rgb", label="fault", positions=val_fault_pos, return_type="pil",
        with_masks=True, mask_return_type="numpy", mask_binarize=True, mask_align="order",
        rgb_policy=rgb_policy
    )

    fault_val_paths = None
    if debug_print_val_paths:
        # stessi fault ma come Path, per sapere i nomi file
        fault_val_paths, _ = get_items(
            part, "rgb", label="fault", positions=val_fault_pos, return_type="path",
            with_masks=True, mask_return_type="path", mask_binarize=True, mask_align="order",
            rgb_policy=rgb_policy
        )
        print(f"\n[debug] FAULT in validation per {part}, posizioni {val_fault_pos}:")
        for p in fault_val_paths:
            print("   ", p.name)

    # --- generatori separati ma compatibili ---
    rng_val = torch.Generator().manual_seed(seed)
    if train_seed is None or train_seed == seed:
        rng_train = rng_val
    else:
        rng_train = torch.Generator().manual_seed(train_seed)

    train_keep_by_pos: dict[str, list[Image.Image]] = {}
    val_sel_by_pos:   dict[str, list[Image.Image]] = {}
    per_pos_counts:   dict[str, dict] = {}

    # NEW: path dei GOOD selezionati per validation, per posizione
    val_sel_paths_by_pos: dict[str, list[Path]] = {}

    # 1) posizioni del TRAIN: splitto togliendo k per la val (usando rng_val)
    for pos in train_pos_list:
        # immagini come PIL
        imgs_pos = get_items(
            part, "rgb", label="good", positions=[pos], return_type="pil",
            rgb_policy=rgb_policy
        )
        n = len(imgs_pos)

        raw = float(k_map_raw.get(pos, 0.0)) if pos in val_good_pos else 0.0
        if raw <= 0.0:
            k = 0
        else:
            if raw < 1.0:
                k = int(round(raw * n))
            else:
                k = int(raw)
            k = _cap_warn(pos, k, n)

        if k > 0:
            idx = torch.randperm(n, generator=rng_val).tolist()
            sel  = [imgs_pos[i] for i in idx[:k]]
            keep = [imgs_pos[i] for i in idx[k:]]

            if debug_print_val_paths:
                # stessi file come Path, in ordine deterministico
                paths_pos = get_items(
                    part, "rgb", label="good", positions=[pos], return_type="path",
                    rgb_policy=rgb_policy
                )
                sel_paths = [paths_pos[i] for i in idx[:k]]
                val_sel_paths_by_pos[pos] = sel_paths

                print(f"\n[debug] GOOD in validation per {part}/{pos} (da TRAIN):")
                for p in sel_paths:
                    print("   ", p.name)
        else:
            sel, keep = [], imgs_pos

        val_sel_by_pos[pos]    = sel
        train_keep_by_pos[pos] = keep
        per_pos_counts[pos] = {
            "good_total": n,
            "good_val":   len(sel),
            "good_train": len(keep),
        }

    # 2) posizioni extra per la validation (non in train), sempre con rng_val
    extra_val_pos = [p for p in val_good_pos if p not in train_pos_list]
    for pos in extra_val_pos:
        imgs_pos = get_items(
            part, "rgb", label="good", positions=[pos], return_type="pil",
            rgb_policy=rgb_policy
        )
        n = len(imgs_pos)
        raw = float(k_map_raw.get(pos, 0.0))
        if raw <= 0.0:
            k = 0
        else:
            if raw < 1.0:
                k = int(round(raw * n))
            else:
                k = int(raw)
            k = _cap_warn(pos, k, n)

        if k > 0:
            idx = torch.randperm(n, generator=rng_val).tolist()
            sel = [imgs_pos[i] for i in idx[:k]]

            if debug_print_val_paths:
                paths_pos = get_items(
                    part, "rgb", label="good", positions=[pos], return_type="path",
                    rgb_policy=rgb_policy
                )
                sel_paths = [paths_pos[i] for i in idx[:k]]
                val_sel_paths_by_pos[pos] = sel_paths

                print(f"\n[debug] GOOD in validation per {part}/{pos} (solo VAL):")
                for p in sel_paths:
                    print("   ", p.name)
        else:
            sel = []

        val_sel_by_pos[pos] = sel
        per_pos_counts[pos] = {"good_total": n, "good_val": len(sel), "good_train": 0}

    # --- sotto-campionamento percentuale del TRAIN per posizione (rng_train) ---
    good_train_remaining: list[Image.Image] = []

    for pos, keep in train_keep_by_pos.items():
        frac_pos = _get_frac_for_pos(good_fraction, pos)
        if frac_pos >= 0.999:
            picked = keep
        else:
            n_keep = len(keep)
            n_pick = max(0, int(round(n_keep * frac_pos)))
            if n_pick < n_keep:
                idx = torch.randperm(n_keep, generator=rng_train).tolist()
                picked = [keep[i] for i in idx[:n_pick]]
            else:
                picked = keep
        good_train_remaining.extend(picked)
        per_pos_counts[pos]["good_train_after_fraction"] = len(picked)

    # per posizioni che stanno solo in val (nessun train)
    for pos in extra_val_pos:
        per_pos_counts[pos].setdefault("good_train_after_fraction", 0)

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

    # mappa effettiva dei good spostati in validation per pos
    val_good_counts = {pos: len(sel) for pos, sel in val_sel_by_pos.items()}

    # NEW: nomi file GOOD di validation per posizione
    val_good_files = {
        pos: [p.name for p in paths]
        for pos, paths in val_sel_paths_by_pos.items()
    }

    meta = {
        "train_tag": train_tag,
        "good_fraction": good_fraction,
        "train_positions": train_pos_list,
        "val_fault_positions": val_fault_pos,
        "val_good_scope": val_good_scope,
        "val_good_positions": val_good_pos,
        "val_good_requested": k_map_raw,
        "val_good_per_pos": val_good_counts,
        "counts": {
            "train_good": len(good_train_ds),
            "val_good": len(good_val_selected),
            "val_fault": len(fault_val_ds),
            "val_total": (len(good_val_selected) + len(fault_val_ds)),
        },
        "per_pos_counts": per_pos_counts,
        "val_good_files": val_good_files,                    # <<< NEW
        "val_fault_files": [p.name for p in fault_val_paths] if fault_val_paths is not None else None,
    }
    return good_train_ds, val_set, meta

# ---------------- loaders ----------------
def make_loaders(train_set, val_set, batch_size=32, device: str | torch.device = "cpu"):
    pin = (str(device) == "cuda") or (isinstance(device, torch.device) and device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader
