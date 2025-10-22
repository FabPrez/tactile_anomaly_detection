# Memory_bank_AD/downscale_simple_all_cv2.py
from pathlib import Path
import numpy as np
import cv2

# === CONFIG ===
DATASET_ROOT = Path(__file__).resolve().parent / "dataset"  # .../Memory_bank_AD/dataset
SETS = ["PZ3"]
SRC_W, SRC_H = 1280, 720
TGT_W, TGT_H = 320, 240
POOL_W, POOL_H = SRC_W // TGT_W, SRC_H // TGT_H  # 4, 3
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def resize_rgb_cv2(img_bgr: np.ndarray) -> np.ndarray:
    """
    Downscale area-like per le RGB (equivalente a PIL.Image.BOX).
    img_bgr: HxWx3 BGR uint8
    """
    return cv2.resize(img_bgr, (TGT_W, TGT_H), interpolation=cv2.INTER_AREA)

def maxpool_mask(arr: np.ndarray) -> np.ndarray:
    """
    Max-pooling 3x4 sulle maschere binarie, identico alla tua versione:
    - binarizza ( > 0 )
    - reshape in blocchi 3x4
    - max per blocco
    - rimappa 0/255
    Richiede maschera 1280x720.
    """
    m = (arr > 0).astype(np.uint8)
    H, W = m.shape
    if (W, H) != (SRC_W, SRC_H):
        raise ValueError(f"mask size {W}x{H}, expected {SRC_W}x{SRC_H}")
    m = m.reshape(H // POOL_H, POOL_H, W // POOL_W, POOL_W)
    pooled = m.max(axis=(1, 3)).astype(np.uint8) * 255
    return pooled

def process_rgb_dir(src_dir: Path, out_dir: Path, label: str):
    """Downscale ricorsivo delle RGB preservando la struttura."""
    if not src_dir.exists():
        print(f"[INFO] Skip {label}: {src_dir} non esiste")
        return 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.rglob("*") if is_img(p))
    ok = err = 0
    for p in files:
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR
            if img is None:
                raise ValueError("imread ha restituito None (file corrotto o formato non supportato)")
            img_ds = resize_rgb_cv2(img)
            rel = p.relative_to(src_dir)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # salviamo in PNG per uniformità, come nel tuo “force png if needed”
            out_path = dst.with_suffix(".png") if dst.suffix.lower() not in IMG_EXTS else dst
            cv2.imwrite(str(out_path), img_ds)
            ok += 1
        except Exception as e:
            print(f"[WARN] RGB KO ({label}): {p} -> {e}")
            err += 1
    print(f"[RGB {label:>5}] {ok}/{len(files)} salvate → {out_dir}")
    return ok, err

def process_masks(pos_dir: Path):
    mdir = pos_dir / "fault" / "mask"
    out_dir = pos_dir / "fault" / "mask_320x240"
    if not mdir.exists():
        print(f"[INFO] Skip masks: {mdir} non esiste")
        return 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in mdir.rglob("*") if is_img(p))
    ok = skip = 0
    for p in files:
        try:
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # 2D uint8, EXIF ignorato (coerente con cv2)
            if arr is None:
                raise ValueError("imread ha restituito None (file corrotto o formato non supportato)")
            pooled = maxpool_mask(arr)  # SOLO max-pooling 3x4 (stessa semantica)
            rel = p.relative_to(mdir)
            out_path = (out_dir / rel).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), pooled)
            ok += 1
        except Exception as e:
            print(f"[WARN] MASK SKIP: {p} -> {e}")
            skip += 1
    print(f"[MASK ] ok={ok}, skip={skip} → {out_dir}")
    return ok, skip

def process_set(set_name: str):
    root = DATASET_ROOT / set_name
    if not root.exists():
        print(f"[INFO] Skip {set_name}: {root} non esiste")
        return
    for pos_dir in sorted(p for p in root.glob("pos*") if p.is_dir()):
        # RGB good: posX/rgb -> posX/rgb_320x240
        process_rgb_dir(pos_dir / "rgb", pos_dir / "rgb_320x240", label="good")
        # RGB fault: posX/fault/rgb -> posX/fault/rgb_320x240
        process_rgb_dir(pos_dir / "fault" / "rgb", pos_dir / "fault" / "rgb_320x240", label="fault")
        # MASK fault: posX/fault/mask -> posX/fault/mask_320x240 (max-pooling)
        process_masks(pos_dir)

if __name__ == "__main__":
    print("DATASET_ROOT =", DATASET_ROOT.resolve())
    for s in SETS:
        process_set(s)
