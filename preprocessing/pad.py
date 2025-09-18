from PIL import Image, ImageOps

def preprocess(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    width, height = img.size
    delta_w = max(224 - width, 0)
    delta_h = max(224 - height, 0)
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding, fill=0)
    return img.resize((224, 224), Image.BILINEAR)
