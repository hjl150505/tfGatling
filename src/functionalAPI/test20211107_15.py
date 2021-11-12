import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
函数式api创建模型与子类化Model的区别：
1、没有 super(MyClass, self).__init__(...)，没有 def call(self, ...): 等内容。
2、定义连接计算图时进行模型验证
3、函数式模型可绘制且可检查
4、函数式模型可以序列化或克隆
    要序列化子类化模型，实现器必须在模型级别指定 get_config() 和 from_config() 方法。
5、不支持动态架构
"""

inputs = keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
mlp = keras.Model(inputs, outputs)

class MLP(keras.Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(64, activation='relu')
        self.dense_2 = layers.Dense(10)
    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)
# Instantiate the model.
mlp = MLP()
# Necessary to create the model's state.
# The model doesn't have a state until it's called at least once.
_ = mlp(tf.zeros((1, 32)))