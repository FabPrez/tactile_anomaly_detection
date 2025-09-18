from PIL import Image
from preprocessing.pad import preprocess  # Cambia qui il metodo da testare
import os

folder = "C:/Users/matilde/Desktop/PROGETTI/tactile_anomaly_detection/immagini"
print("File presenti nella cartella immagini:")
print(os.listdir(folder))

# Percorso corretto del file immagine (non della cartella)
img_path = "C:/Users/matilde/Desktop/PROGETTI/tactile_anomaly_detection/immagini/image1.PNG"

img = Image.open(img_path)
img_proc = preprocess(img)
print("Dimensione immagine processata:", img_proc.size)
img_proc.show()  # Mostra l'immagine processata
# img_proc.save("immagine_processata.jpg")  # Salva se vuoi
