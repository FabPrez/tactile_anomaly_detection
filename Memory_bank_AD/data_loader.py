# dataset_index.py
from __future__ import annotations

from pathlib import Path
from typing import List, Iterable, Union, Literal, Optional
import pickle
import re

from PIL import Image
import numpy as np

# Base del dataset: cartella "Dataset" accanto a questo file
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "Dataset"

# Estensioni supportate
EXTS = {
    "rgb": {".png"},          # puoi aggiungere altre estensioni se servono
    "pointcloud": {".ply"},
}
EXTS_MASK = {".png"}          # maschere (binary png)

# ---------------- natural sort ----------------

_num_re = re.compile(r"(\d+)")

def _natsort_key(s: str):
    """Chiave per 'natural sort': 'task-10' viene dopo 'task-2'."""
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(s)]

def _iter_files(
    root: Path,
    exts: set[str],
    sort_mode: Literal["natural", "name", "mtime"] = "natural",
) -> List[Path]:
    """Ritorna i file con estensioni valide sotto root (ricorsivo), ordinati."""
    if not root.exists():
        return []
    exts_low = {e.lower() for e in exts}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_low]

    if sort_mode == "natural":
        files.sort(key=lambda p: _natsort_key(p.name))
    elif sort_mode == "name":
        files.sort(key=lambda p: p.name.lower())
    elif sort_mode == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)  # dalla più vecchia alla più nuova
    else:
        files.sort(key=lambda p: p.name.lower())
    return files

# -------- util --------

def list_positions(part: str) -> List[str]:
    """Ritorna la lista delle posizioni disponibili (es. ['pos1','pos2',...])."""
    base = DATASET_ROOT / part
    if not base.exists():
        raise FileNotFoundError(f"Part non trovato: {base}")
    return sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.lower().startswith("pos"))

def _normalize_positions(part: str, positions: Union[None, str, Iterable[str]]) -> List[str]:
    """Normalizza il parametro positions in una lista di posizioni."""
    if positions is None or positions == "*" or positions == "all":
        return list_positions(part)
    if isinstance(positions, str):
        return [positions]
    return list(positions)

def _load_rgb_pil(path: Path) -> Image.Image:
    """Apre un'immagine RGB da path come PIL.Image (gestendo EXIF orientation)."""
    img = Image.open(path).convert("RGB")
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img

def _load_mask_pil(path: Path) -> Image.Image:
    """Maschera come L (8-bit, 0..255)."""
    img = Image.open(path).convert("L")
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img

def _norm_stem_for_match(stem: str) -> str:
    """Normalizza lo stem per abbinare immagine e mask se si usa mask_align='name'."""
    s = stem.lower()
    for suf in ("_mask", "-mask", ".mask", " mask"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

# -------- API principali --------

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
    mask_align: Literal["order", "name"] = "order",  # "order" = per indice (dopo natural sort)
) -> Union[
    List[Union[Path, Image.Image, np.ndarray]],
    tuple[List[Union[Path, Image.Image, np.ndarray]], List[Optional[Union[Path, Image.Image, np.ndarray]]]]
]:
    """
    Restituisce elementi per:
      - part: 'PZ1' | 'PZ2' | ...
      - modality: 'rgb' | 'pointcloud'
      - label: 'good' | 'fault'
      - positions: None/'all'/'*' oppure 'pos2' oppure iterable di pos

    return_type:
      - 'path'  -> lista di pathlib.Path
      - 'pil'   -> lista di PIL.Image.Image (solo modality='rgb')
      - 'numpy' -> lista di np.ndarray (H,W,C) uint8 (solo modality='rgb')

    Opzioni mask:
      - with_masks=True -> ritorna anche la lista maschere allineata (solo fault/rgb).
      - mask_align="order": abbina per ordine (natural sort) — CONSIGLIATO per il tuo caso.
      - mask_align="name" : abbina per nome normalizzato.
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
    mask_paths_all: List[Optional[Path]] = []  # allineata a results_paths (None se assente)

    for pos in pos_list:
        # ----- immagini -----
        img_dir = (base / pos / modality) if label == "good" else (base / pos / "fault" / modality)
        imgs_here = _iter_files(img_dir, EXTS[modality], sort_mode="natural")

        if not img_dir.exists():
            print(f"[!] Cartella mancante: {img_dir}")
        else:
            total = len(list(img_dir.glob("*")))
            print(f"[{part}/{pos}/{label}/{modality}] Trovati {len(imgs_here)} di {total} file")

        results_paths.extend(imgs_here)

        # ----- maschere (solo se richieste e fault/rgb) -----
        if with_masks and (label == "fault") and (modality == "rgb"):
            mask_dir = base / pos / "fault" / "mask"
            masks_here = _iter_files(mask_dir, EXTS_MASK, sort_mode="natural") if mask_dir.exists() else []

            if mask_align == "order":
                # Abbina per indice (dopo natural sort)
                if len(masks_here) != len(imgs_here):
                    print(f"[warn] {part}/{pos}: #mask ({len(masks_here)}) != #img ({len(imgs_here)}) — allineo per indice con padding/troncamento.")
                for i in range(len(imgs_here)):
                    mp = masks_here[i] if i < len(masks_here) else None
                    mask_paths_all.append(mp)

            elif mask_align == "name":
                # Abbina per nome normalizzato
                mask_map = {_norm_stem_for_match(mp.stem): mp for mp in masks_here}
                for ip in imgs_here:
                    key = _norm_stem_for_match(ip.stem)
                    mp = mask_map.get(key, None)
                    if mp is None:
                        # fallback con suffissi comuni
                        for suf in ("_mask", "-mask"):
                            alt = (mask_dir / f"{ip.stem}{suf}{ip.suffix}")
                            if alt.exists():
                                mp = alt
                                break
                        if mp is None:
                            print(f"[warn] mask non trovata per {ip.name} in {mask_dir}")
                    mask_paths_all.append(mp)
            else:
                raise ValueError("mask_align deve essere 'order' o 'name'.")
        elif with_masks:
            # richieste mask ma non applicabili: riempi con None
            mask_paths_all.extend([None] * len(imgs_here))

    # ---- solo immagini (comportamento classico) ----
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

    # ---- con maschere: converto immagini + maschere ----
    # immagini
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

    # maschere
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

# -------- util pickle --------

def _features_dir_for(part: str, position: str | None) -> Path:
    """
    Se position è None/'all'/'*' -> Dataset/<part>/features
    Altrimenti -> Dataset/<part>/<position>/features
    """
    pos_norm = None if position is None else str(position).lower()
    if pos_norm in (None, "all", "*"):
        return DATASET_ROOT / part / "features"
    else:
        return DATASET_ROOT / part / position / "features"


def _normalize_method(method: str) -> str:
    """
    Normalizza il nome del metodo per l'uso nel filename:
    - lowercase
    - spazi -> '-'
    - rimuove caratteri non alfanumerici/underscore/dash
    """
    m = method.strip().lower().replace(" ", "-")
    m = re.sub(r"[^a-z0-9_-]", "", m)
    if not m:
        raise ValueError("method non può essere vuoto dopo la normalizzazione")
    return m


def save_split_pickle(obj, part: str, position: str | None, split: str, method: str) -> Path:
    """
    Salva un pickle in:
      - Dataset/<part>/features/<method>_<split>.pickle
      - Dataset/<part>/<position>/features/<method>_<split>.pickle
    split: 'train' | 'validation'
    method: nome del metodo (es. 'spade', 'padim', ...)
    """
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
    """
    Carica un pickle da:
      - Dataset/<part>/features/<method>_<split>.pickle
      - Dataset/<part>/<position>/features/<method>_<split>.pickle
    split: 'train' | 'validation'
    method: nome del metodo (es. 'spade', 'padim', ...)
    """
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