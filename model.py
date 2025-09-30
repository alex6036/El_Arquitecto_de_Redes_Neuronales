import tensorflow as tf
import numpy as np

class DigitClassifier:
    def __init__(self, model_path="mnist_cnn.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img_array):
        """
        img_array: numpy array (28x28x1)
        devuelve el dígito predicho
        """
        img_array = img_array.reshape(1, 28, 28, 1)  # añadir batch
        predictions = self.model.predict(img_array)
        return np.argmax(predictions, axis=1)[0]

