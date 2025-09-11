from pathlib import Path
from typing import Iterable, List, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _iter_imgs(root: Path):
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)

def list_images_by_part(
    dataset_root: str | Path,
    part: str,
    positions: Iterable[str] | None = None,
    modality: str = "rgb",
) -> Tuple[List[Path], List[Path]]:
    """
    Restituisce (good_paths, defect_paths) per un pezzo (PZ1, PZ2...).

    Struttura attesa:
      Dataset/PZx/posK/rgb/            -> good
      Dataset/PZx/posK/fault/rgb/      -> defect
    """
    dataset_root = Path(dataset_root)
    base = dataset_root / part
    # Crea la struttura minima se non esiste
    if not base.exists():
        print(f"Creo la struttura {base/'pos1/rgb'} e {base/'pos1/fault/rgb'}")
        (base / "pos1" / "rgb").mkdir(parents=True, exist_ok=True)
        (base / "pos1" / "fault" / "rgb").mkdir(parents=True, exist_ok=True)

    if positions is None:
        positions = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("pos"))

    good, defect = [], []
    for pos in positions:
        pos_dir = base / pos
        good_dir   = pos_dir / modality
        defect_dir = pos_dir / "fault" / modality

        good_imgs = sorted(_iter_imgs(good_dir))
        defect_imgs = sorted(_iter_imgs(defect_dir))

        if not good_imgs:
            print(f"Attenzione: nessuna immagine trovata in {good_dir}")
        if not defect_imgs:
            print(f"Attenzione: nessuna immagine trovata in {defect_dir}")

        good.extend(good_imgs)
        defect.extend(defect_imgs)

    return good, defect
