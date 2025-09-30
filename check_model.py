import tensorflow as tf

model = tf.keras.models.load_model("digit_model.h5")
model.summary()
