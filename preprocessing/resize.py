from PIL import Image

def preprocess(img):
    # img può essere PIL.Image o numpy array
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img.resize((224, 224), Image.BILINEAR)
