import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

class DigitClassifier:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        return model

    def train(self, x_train, y_train, x_test, y_test, epochs=3, batch_size=128):
        if self.model is None:
            self.build_model()
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def predict(self, image):
        image = image.reshape(1, 28, 28, 1)  # batch de 1
        prediction = self.model.predict(image)
        return prediction.argmax()

    def save(self, path="digit_model.h5"):
        self.model.save(path)

    def load(self, path="digit_model.h5"):
        self.model = load_model(path)
        return self.model
