from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class LetterboxToSquare:
    """
    Ridimensiona mantenendo l'aspect ratio (lato lungo -> target)
    e aggiunge padding simmetrico per ottenere un quadrato target x target.
    """
    def __init__(self, target: int = 224, fill=(0, 0, 0)):
        self.target = target
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Immagine con dimensioni non valide.")
        if w < 10 or h < 10:
            print("Attenzione: immagine molto piccola, potrebbero esserci problemi di qualità.")
        scale = self.target / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.target - new_w
        pad_h = self.target - new_h
        if pad_w < 0 or pad_h < 0:
            print("Attenzione: padding negativo, controlla la dimensione target.")
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)  # L,T,R,B
        img = TF.pad(img, padding, fill=self.fill)
        return img

def build_resnet_preprocess(
    target_size: int = 224,
    mode: str = "letterbox",  # "letterbox" oppure "center_crop"
):
    """
    Ritorna una Compose che produce tensori 3x[target_size]x[target_size] normalizzati ImageNet.
    - 'letterbox': nessun taglio, padding a quadrato (consigliato)
    - 'center_crop': Resize(256) + CenterCrop(target_size), come ImageNet
    """
    if mode == "letterbox":
        return T.Compose([
            LetterboxToSquare(target=target_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    elif mode == "center_crop":
        return T.Compose([
            T.Resize(target_size + 32, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        raise ValueError("mode deve essere 'letterbox' o 'center_crop'")
