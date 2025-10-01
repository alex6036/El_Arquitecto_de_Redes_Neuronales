# main.py
# Punto de entrada para el proyecto de redes neuronales
import numpy as np
from src.classes.mlp import MLP

if __name__ == "__main__":
    # Dataset XOR
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])  # etiquetas reales

    # Configuración del MLP: [ (inputs, neuronas, activación) ... ]
    config = [
        (2, 4, "relu"),      # capa oculta 1
        (4, 4, "relu"),      # capa oculta 2
        (4, 1, "sigmoid")    # capa salida
    ]

    # Crear el MLP
    mlp = MLP(config)

    # Predicciones iniciales
    preds = mlp.predict(X)
    print("Predicciones iniciales (sin entrenar):")
    print(preds)
