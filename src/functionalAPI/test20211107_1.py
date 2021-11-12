import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
创建模型，并画图，画模型结构
"""
inputs = keras.Input(shape=(784,))
print(inputs.shape)
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

print(model.summary())
keras.utils.plot_model(model, "my_first_model.png")
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
