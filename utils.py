import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocesa la imagen para que se parezca a las de MNIST:
    - Convierte a gris
    - Invierte colores
    - Binariza fuerte
    - Recorta el dígito
    - Redimensiona y centra en 28x28
    """
    # Escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invertir: fondo negro, dígito blanco
    gray = 255 - gray

    # Binarización fuerte
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Buscar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        digit = cv2.resize(binary, (28, 28))
    else:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        digit = binary[y:y+h, x:x+w]

        # Redimensionar manteniendo proporción a 20x20
        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = int(w * (20.0 / h))
        else:
            new_w = 20
            new_h = int(h * (20.0 / w))
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Poner en un canvas 28x28
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit
        digit = canvas

    # Normalizar y añadir canal
    digit = digit.astype(np.float32) / 255.0
    digit = np.expand_dims(digit, axis=-1)

    # Debug
    cv2.imwrite("debug_digit.png", (digit.squeeze() * 255).astype(np.uint8))
    print("DEBUG - Imagen procesada guardada como debug_digit.png")
    return digit


