import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
提取层计算图中的节点
提取中间层特征数据
使用已经训练好的模型的某些层进行预测
"""

vgg19 = tf.keras.applications.VGG19()
features_list = [layer.output for layer in vgg19.layers]
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)
print(extracted_features)