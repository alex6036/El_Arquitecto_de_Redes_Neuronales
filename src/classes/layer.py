# layer.py
# Definición de capa para redes neuronales
import numpy as np
from src.functions.neuron import neuron

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "sigmoid"):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((n_neurons,))
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante para un lote de entradas.
        inputs: shape (batch_size, n_inputs)
        return: shape (batch_size, n_neurons)
        """
        batch_outputs = []
        for x in inputs:
            outputs = []
            for j in range(self.weights.shape[1]):
                outputs.append(
                    neuron(x, self.weights[:, j], self.biases[j], self.activation)
                )
            batch_outputs.append(outputs)
        return np.array(batch_outputs)
