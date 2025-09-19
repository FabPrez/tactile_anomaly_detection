from PIL import Image

def preprocess(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.resize((256, 256), Image.BILINEAR)
    width, height = img.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = left + 224
    bottom = top + 224
    return img.crop((left, top, right, bottom))
