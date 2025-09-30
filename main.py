import gradio as gr
from model import DigitClassifier
from utils import preprocess_image

def classify_image_gradio(image):
    """
    Función que recibe imagen de Gradio y devuelve número predicho
    """
    img_array = preprocess_image(image)
    classifier = DigitClassifier()
    digit = classifier.predict(img_array)
    return digit

if __name__ == "__main__":
    interface = gr.Interface(
        fn=classify_image_gradio,
        inputs=gr.Image(type="numpy"),
        outputs=gr.Textbox(label="Número predicho"),
        title="Clasificador de Números MNIST",
        description="Sube una imagen de un dígito manuscrito (0-9) y el modelo te dirá qué número es."
    )

    interface.launch()

