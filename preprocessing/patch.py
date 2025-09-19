from PIL import Image
import random

def preprocess(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    width, height = img.size
    if width < 224 or height < 224:
        img = img.resize((max(224, width), max(224, height)), Image.BILINEAR)
        width, height = img.size
    left = random.randint(0, width - 224)
    top = random.randint(0, height - 224)
    right = left + 224
    bottom = top + 224
    return img.crop((left, top, right, bottom))
