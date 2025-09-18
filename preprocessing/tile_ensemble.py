from PIL import Image
import numpy as np

def preprocess(img, tile_size=224, stride=112):
    """
    Divide l'immagine in tile sovrapposti di dimensione tile_size x tile_size.
    Restituisce una lista di tile (PIL.Image).
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    width, height = img.size
    tiles = []
    for top in range(0, height - tile_size + 1, stride):
        for left in range(0, width - tile_size + 1, stride):
            box = (left, top, left + tile_size, top + tile_size)
            tile = img.crop(box)
            tiles.append(tile)
    # Gestione dei bordi: aggiungi tile anche se non perfettamente divisibile
    if (height - tile_size) % stride != 0:
        for left in range(0, width - tile_size + 1, stride):
            box = (left, height - tile_size, left + tile_size, height)
            tile = img.crop(box)
            tiles.append(tile)
    if (width - tile_size) % stride != 0:
        for top in range(0, height - tile_size + 1, stride):
            box = (width - tile_size, top, width, top + tile_size)
            tile = img.crop(box)
            tiles.append(tile)
    # Angolo in basso a destra se necessario
    if (height - tile_size) % stride != 0 and (width - tile_size) % stride != 0:
        box = (width - tile_size, height - tile_size, width, height)
        tile = img.crop(box)
        tiles.append(tile)
    return tiles
