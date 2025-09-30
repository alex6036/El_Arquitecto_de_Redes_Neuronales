import cv2
import numpy as np

def preprocess_image(image):
    """
    image: array de imagen subido por Gradio (RGB)
    devuelve array 28x28x1 normalizado para el modelo
    """
    # Convertir a escala de grises
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Redimensionar a 28x28
    img = cv2.resize(img, (28, 28))
    # Normalizar
    img = img / 255.0
    # Invertir colores si es necesario (MNIST: fondo negro, número blanco)
    img = 1 - img
    img = img.astype(np.float32)
    # Añadir canal
    img = np.expand_dims(img, axis=-1)
    return img

