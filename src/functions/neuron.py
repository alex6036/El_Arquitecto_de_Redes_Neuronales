# neuron.py
# Definición de neurona para redes neuronales
import numpy as np
from src.functions.activations import sigmoid, relu

def neuron(x: np.ndarray, weights: np.ndarray, bias: float, activation: str = "sigmoid") -> float:
    """
    Simula una neurona.
    x: entradas (vector)
    weights: pesos
    bias: sesgo
    activation: 'sigmoid' o 'relu'
    """
    z = np.dot(x, weights) + bias
    
    if activation == "sigmoid":
        return sigmoid(z)
    elif activation == "relu":
        return relu(z)
    else:
        raise ValueError(f"Función de activación '{activation}' no soportada.")
