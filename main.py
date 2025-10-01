import gradio as gr
import cv2
import os
import numpy as np
from model import DigitClassifier
from utils import preprocess_image

SAVE_DIR = "my_digits"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------- CLASIFICADOR ----------
def classify_image_gradio(image):
    """
    Recibe imagen (numpy) de Gradio y devuelve número predicho
    """
    img_array = preprocess_image(image)
    classifier = DigitClassifier()
    digit = classifier.predict(img_array)
    return digit


# ---------- ENTRENADOR ----------
def save_digit(image, label):
    """
    Guarda el dígito dibujado con su etiqueta en la carpeta my_digits
    """
    if image is None:
        return "❌ No hay imagen"

    # Extraer la imagen si viene como diccionario desde Gradio
    if isinstance(image, dict) and "image" in image:
        image = image["image"]

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Redimensionar a 28x28
    digit = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Guardar con nombre único
    filename = os.path.join(SAVE_DIR, f"{label}_{len(os.listdir(SAVE_DIR))}.png")
    cv2.imwrite(filename, digit)

    return f"✅ Imagen guardada como {filename}"



# ---------- MAIN ----------
if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## ✨ Proyecto: Clasificador y Entrenador de Números MNIST")

        with gr.Tab("🔢 Clasificador"):
            gr.Markdown("Sube una imagen y el modelo te dirá qué número es.")
            with gr.Row():
                img_input = gr.Image(type="numpy", label="Sube tu dígito")
                output = gr.Textbox(label="Número predicho")
            img_input.change(fn=classify_image_gradio, inputs=img_input, outputs=output)

        with gr.Tab("🖌️ Entrenador"):
            gr.Markdown("Dibuja un número, selecciona su etiqueta y guárdalo para entrenar después.")
            with gr.Row():
                sketch = gr.Sketchpad(canvas_size=(200, 200), type="numpy")

                label = gr.Dropdown(choices=[str(i) for i in range(10)], label="Número")
            save_btn = gr.Button("💾 Guardar")
            save_output = gr.Textbox(label="Estado")
            save_btn.click(fn=save_digit, inputs=[sketch, label], outputs=save_output)

    demo.launch()
