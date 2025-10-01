# main.py
# Punto de entrada para el proyecto de redes neuronales
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from src.compiler.model_compiler import compile_model

def main():
    # -----------------------------
    # 1️⃣ Cargar y preparar MNIST
    # -----------------------------
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    # Normalizar imágenes (0-1)
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    # Aplanar imágenes (28x28 -> 784)
    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)

    # Convertir etiquetas a one-hot
    trainY_cate = to_categorical(trainY, num_classes=10)
    testY_cate = to_categorical(testY, num_classes=10)

    print(f"TrainX shape: {trainX.shape}, TrainY shape: {trainY_cate.shape}")
    print(f"TestX shape: {testX.shape}, TestY shape: {testY_cate.shape}")

    # -----------------------------------
    # 2️⃣ Compilar la arquitectura
    # -----------------------------------
    architecture = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    input_dim = trainX.shape[1]  # 784
    model = compile_model(architecture, input_dim=input_dim)

    # -----------------------------
    # 3️⃣ Entrenar el modelo
    # -----------------------------
    model.fit(
        trainX, trainY_cate,
        validation_data=(testX, testY_cate),
        epochs=5,
        batch_size=128,
        verbose=2
    )

    # -----------------------------
    # 4️⃣ Evaluar rendimiento
    # -----------------------------
    loss, accuracy = model.evaluate(testX, testY_cate, verbose=0)
    print(f"\nTest accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
