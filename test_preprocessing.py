from PIL import Image
from preprocessing.resize import preprocess  # Cambia qui il metodo da testare

# Percorso corretto dell'immagine nella cartella 'immagini' del progetto
img_path = "C:/Users/matilde/Desktop/PROGETTI/tactile_anomaly_detection/immagini/image1.jpg""

img = Image.open(img_path)
img_proc = preprocess(img)
print("Dimensione immagine processata:", img_proc.size)
img_proc.show()  # Mostra l'immagine processata
# img_proc.save("immagine_processata.jpg")  # Salva se vuoi
