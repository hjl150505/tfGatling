import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from collections import defaultdict
from itertools import chain
from tensorflow.python.keras.layers import LSTM, Lambda, Layer
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l1_l2

from tfDataSet import v23_1_1
from parseFeatSchema import v23_1_1parserTsv

"""
测试embedding层的使用
python=3.6
tensorflow=2.3.0

"""


class ModelDin():
    def __init__(self):
        self.hidden_units = (80, 40)
        self.seed = 1024
        self.hist_mask_value = 0
        self.l2_reg = 0
        self.use_bn = False
        self.dropout_rate = 0
        self.activation = 'dice'
        self.output_activation = None
        self.weight_normalization = False
        self.return_score = False
        self.att_activation = 'sigmoid'
        self.feat_props, self.num_feat, self.cat_feat = v23_1_1.getFeatParserInfo()

    def buildInput(self):
        input_features = {}
        sparse_embedding = {}
        l2_reg = 1e-6
        for use, name, size, buckets, weight_fc, default, dtype in self.cat_feat:
            if dtype != tf.int64 or name == 'userId':
                continue
            shape = (size,)
            mask_flag = True
            input_features[name] = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            cur_emb = Embedding(buckets, 10,
                                embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2022),
                                embeddings_regularizer=l2(l2_reg),
                                name='sparse_emb_' + name,
                                mask_zero=mask_flag)
            cur_emb.trainable = True
            sparse_embedding[name] = cur_emb

        for use, name, size, mean, stddev, normalize, default, dtype, in self.num_feat:
            if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight':
                continue
            shape = (size,)
            input_features[name] = keras.layers.Input(name=name, shape=shape, dtype=dtype)

        return input_features, sparse_embedding

    def build(self):
        hidden_units = [32, 16]
        dropout = 0.2
        input_features, embDic = self.buildInput()


if __name__ == "__main__":
    x = v23_1_1.data_gen(["F:\\data\\tensorflow\\v14_5_8\\date=20221115\\train\\part-r-00018",
                          "F:\\data\\tensorflow\\v14_5_8\\date=20221115\\train\\part-r-00019"], 4)
    v = v23_1_1.data_gen("F:\\data\\tensorflow\\v14_5_8\\date=20221115\\val\\part-r-00005", 4)
    t = v23_1_1.data_gen("F:\\data\\tensorflow\\v14_5_8\\date=20221115\\test\\part-r-00003", 4)

    test_op = tf.compat.v1.data.make_one_shot_iterator(x)
    one_element = test_op.get_next()
    print(one_element)
    print("one_element=>", one_element)
    embModel = ModelDin()
    model, loss = embModel.build()
    model.summary()
    model.output_names[0] = 'predict_score'
    optimizer = keras.optimizers.Adam()
    metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(),
               keras.metrics.Recall(), keras.metrics.AUC()]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, experimental_run_tf_function=False)
    history = model.fit(x,
                        validation_data=v,
                        epochs=2,
                        steps_per_epoch=1000)
    model.save("v23_1_1_model")

    loadModel = keras.models.load_model("v23_1_1_model")
    preRs = loadModel.predict(t)
    print(preRs)
