import tensorflow as tf

def compile_model(architecture_string: str, input_dim: int = None) -> tf.keras.Sequential:
    """
    Convierte una descripción textual en un modelo Keras Sequential.
    
    Ejemplo:
        "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    """
    model = tf.keras.Sequential()
    
    # Dividir por "->"
    layers = [layer.strip() for layer in architecture_string.split("->")]
    
    for i, layer in enumerate(layers):
        if layer.startswith("Dense"):
            # Extraer parámetros entre paréntesis
            params = layer[layer.find("(")+1:layer.find(")")].split(",")
            units = int(params[0].strip())
            activation = params[1].strip() if len(params) > 1 else None
            
            # Si es la primera capa y se especifica input_dim
            if i == 0 and input_dim is not None:
                model.add(tf.keras.layers.Dense(units, activation=activation, input_shape=(input_dim,)))
            else:
                model.add(tf.keras.layers.Dense(units, activation=activation))
        else:
            raise ValueError(f"Tipo de capa no soportado: {layer}")
    
    # Compilación básica (puedes ajustarlo luego)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
