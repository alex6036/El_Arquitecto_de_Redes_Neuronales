# mlp.py
# Definición de perceptrón multicapa (MLP)
import numpy as np
from src.classes.layer import Layer

class MLP:
    def __init__(self, layers_config):
        """
        layers_config: lista de tuplas [(n_inputs, n_neurons, activation), ...]
        """
        self.layers = []
        for n_inputs, n_neurons, activation in layers_config:
            self.layers.append(Layer(n_inputs, n_neurons, activation))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza un forward pass por todas las capas.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
