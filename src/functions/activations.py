# activations.py
# Funciones de activación para redes neuronales
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Función Sigmoid"""
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """Función ReLU"""
    return np.maximum(0, x)
