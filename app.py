import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.title("Clasificador de Dígitos MNIST")

model = load_model("data/models/mnist_model.h5")

uploaded_file = st.file_uploader("Sube una imagen de un dígito", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28,28))
    image_array = np.array(image).reshape(1,28*28)/255.0
    pred = np.argmax(model.predict(image_array), axis=1)[0]
    st.image(image, caption="Imagen subida", use_column_width=True)
    st.write(f"Dígito predicho: {pred}")
