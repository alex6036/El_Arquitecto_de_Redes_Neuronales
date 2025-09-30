import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = 255 - gray  # invertir: fondo negro, dígito blanco

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        digit = cv2.resize(gray, (28, 28))
    else:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        digit = gray[y:y+h, x:x+w]

        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = int(w * (20.0 / h))
        else:
            new_w = 20
            new_h = int(h * (20.0 / w))
        digit = cv2.resize(digit, (new_w, new_h))

        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit
        digit = canvas

    # Debug: guardar para inspección
    cv2.imwrite("debug_digit.png", digit)
    print("DEBUG - Imagen procesada guardada como debug_digit.png")

    digit = digit.astype(np.float32) / 255.0
    digit = np.expand_dims(digit, axis=-1)
    print("DEBUG - shape:", digit.shape, "min:", digit.min(), "max:", digit.max())
    return digit

