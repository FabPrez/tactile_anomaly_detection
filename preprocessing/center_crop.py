from PIL import Image

def preprocess(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    width, height = img.size
    new_width, new_height = 224, 224
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return img.crop((left, top, right, bottom))
