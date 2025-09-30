from model import DigitClassifier
from preprocessing import load_mnist
from utils import preprocess_image

def main():
    # 1. Cargar datos
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # 2. Crear modelo
    classifier = DigitClassifier()
    classifier.build_model()

    # 3. Entrenar
    print("Entrenando modelo...")
    classifier.train(x_train, y_train, x_test, y_test, epochs=3)

    # 4. Evaluar
    loss, acc = classifier.evaluate(x_test, y_test)
    print(f"Accuracy en test: {acc:.4f}")

    # 5. Guardar modelo entrenado
    classifier.save("digit_model.h5")

    # 6. Cargar imagen externa
    filepath = "numero.png"  # cambia por tu propia imagen
    img = preprocess_image(filepath)

    # 7. Predecir
    pred = classifier.predict(img)
    print(f"El número detectado es: {pred}")

if __name__ == "__main__":
    main()
