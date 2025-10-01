# main.py
# Punto de entrada para el proyecto de redes neuronales
import tensorflow as tf
from src.compiler.model_compiler import compile_model

if __name__ == "__main__":
    # Ejemplo de arquitectura en mini-lenguaje
    architecture = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    
    # Compilar el modelo con input_dim = 784 (ej. imágenes 28x28 aplanadas)
    model = compile_model(architecture, input_dim=784)
    
    # Mostrar resumen
    model.summary()

