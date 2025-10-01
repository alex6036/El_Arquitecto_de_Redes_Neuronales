import gradio as gr
import cv2
import numpy as np
import os

SAVE_DIR = "my_digits"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_digit(image, label):
    """
    Guarda la imagen dibujada junto con la etiqueta
    """
    if image is None:
        return "❌ No hay imagen"

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Redimensionar a 28x28 (como MNIST)
    digit = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Guardar con nombre único
    filename = os.path.join(SAVE_DIR, f"{label}_{len(os.listdir(SAVE_DIR))}.png")
    cv2.imwrite(filename, digit)

    return f"✅ Imagen guardada como {filename}"

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🖌️ Entrenador de dígitos manuscritos")

    with gr.Row():
        sketch = gr.Sketchpad(shape=(200, 200), brush_radius=10, type="numpy")
        label = gr.Dropdown(choices=[str(i) for i in range(10)], label="Número")

    save_btn = gr.Button("💾 Guardar")
    output = gr.Textbox(label="Estado")

    save_btn.click(fn=save_digit, inputs=[sketch, label], outputs=output)

demo.launch()
