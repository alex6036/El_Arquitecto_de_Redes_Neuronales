import cv2
import numpy as np

def preprocess_image(filepath):
    """
    Carga una imagen externa y la prepara para el modelo:
    - Convierte a escala de grises
    - Redimensiona a 28x28
    - Invierte colores si hace falta
    - Normaliza [0,1]
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # invertir colores (fondo blanco, número negro)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, -1)
    return img
