# dataset_index.py
from pathlib import Path
from typing import List, Iterable, Union, Literal
import pickle

from PIL import Image
import numpy as np

# Base del dataset: cartella "Dataset" accanto a questo file
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "Dataset"

# Estensioni supportate per modalità
EXTS = {"rgb": {".png"}, "pointcloud": {".ply"}}

# -------- util --------

def _iter_files(root: Path, exts: set[str]) -> List[Path]:
    """Ritorna i file con estensioni valide sotto root (ricorsivo, case-insensitive)."""
    if not root.exists():
        return []
    exts_lower = {e.lower() for e in exts}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower)

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

# -------- API principali --------

def get_items(
    part: str,
    modality: str,
    label: str = "good",
    positions: Union[None, str, Iterable[str]] = None,
    return_type: Literal["path", "pil", "numpy"] = "path",
) -> List[Union[Path, Image.Image, np.ndarray]]:
    """
    Restituisce elementi per:
      - part: 'PZ1' | 'PZ2' | 'PZ3' | 'PZ4'
      - modality: 'rgb' | 'pointcloud'
      - label: 'good' | 'fault'
      - positions: None/'all'/'*' oppure 'pos2' oppure iterable di pos

    return_type:
      - 'path'  -> lista di pathlib.Path (default)
      - 'pil'   -> lista di PIL.Image.Image (solo modality='rgb')
      - 'numpy' -> lista di np.ndarray (H,W,C) uint8 (solo modality='rgb')

    Nessun pre-processing viene applicato.
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
    for pos in pos_list:
        target = (base / pos / modality) if label == "good" else (base / pos / "fault" / modality)
        found = _iter_files(target, EXTS[modality])

        if not target.exists():
            print(f"[!] Cartella mancante: {target}")
        else:
            total = len(list(target.glob("*")))
            print(f"[{part}/{pos}/{label}/{modality}] Trovati {len(found)} di {total} file nella cartella")

        results_paths.extend(found)

    if return_type == "path":
        return results_paths

    if modality != "rgb":
        raise ValueError("return_type 'pil' o 'numpy' è supportato solo per modality='rgb'.")

    out: List[Union[Image.Image, np.ndarray]] = []
    for p in results_paths:
        pil_img = _load_rgb_pil(p)
        if return_type == "pil":
            out.append(pil_img)
        elif return_type == "numpy":
            out.append(np.array(pil_img))
        else:
            raise ValueError("return_type non valido.")
    return out

def get_pairs_by_part(
    part: str,
    modality: str,
    positions: Union[None, str, Iterable[str]] = None,
):
    """Restituisce (good_list, fault_list) come path per part/modality e posizioni richieste."""
    return (
        get_items(part, modality, label="good", positions=positions, return_type="path"),
        get_items(part, modality, label="fault", positions=positions, return_type="path"),
    )

# -------- pickle util --------

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


def save_split_pickle(obj, part: str, position: str | None, split: str) -> Path:
    """
    Salva un pickle in:
      - Dataset/<part>/features/<split>.pkl            se position in {None,'all','*'}
      - Dataset/<part>/<position>/features/<split>.pkl altrimenti
    split: 'train' | 'validation'
    """
    split = split.lower()
    if split not in {"train", "validation"}:
        raise ValueError("split deve essere 'train' oppure 'validation'")

    dirpath = _features_dir_for(part, position)
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"{split}.pkl"

    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[pickle] Salvato oggetto in {p}")
    return p


def load_split_pickle(part: str, position: str | None, split: str):
    """
    Carica un pickle da:
      - Dataset/<part>/features/<split>.pkl            se position in {None,'all','*'}
      - Dataset/<part>/<position>/features/<split>.pkl altrimenti
    split: 'train' | 'validation'
    """
    split = split.lower()
    if split not in {"train", "validation"}:
        raise ValueError("split deve essere 'train' oppure 'validation'")

    dirpath = _features_dir_for(part, position)
    p = dirpath / f"{split}.pkl"

    # se non esiste, solleva FileNotFoundError (così il tuo except prende)
    if not p.exists():
        raise FileNotFoundError(f"Pickle non trovato: {p}")

    with open(p, "rb") as f:
        obj = pickle.load(f)
    print(f"[pickle] Caricato oggetto da {p}")
    return obj


