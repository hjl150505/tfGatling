import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import keras_preprocessing
from tensorflow.keras.layers import
input_data = np.array([[1.], [2.], [3.]], dtype='float32')
layer = Normalization(mean=3., variance=2.)
layer(input_data)