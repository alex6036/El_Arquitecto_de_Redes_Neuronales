import os
import tensorflow as tf
import numpy as np

class DigitClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            # Ruta absoluta al archivo en el mismo directorio
            model_path = os.path.join(os.path.dirname(__file__), "digit_model.h5")
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img_array):
        img_array = img_array.reshape(1, 28, 28, 1)
        predictions = self.model.predict(img_array)
        return int(np.argmax(predictions, axis=1)[0])


