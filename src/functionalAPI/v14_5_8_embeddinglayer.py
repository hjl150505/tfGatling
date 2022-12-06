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

from tfDataSet import v14_5_8
from parseFeatSchema import v14_5_8parserTsv

"""
测试embedding层的使用
python=3.6
tensorflow=2.3.0
训练成功
加载模型成功
"""


class testEmb():
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
        self.feat_props, self.num_feat, self.cat_feat = v14_5_8.getFeatParserInfo()

    def buildInput(self):

        # input_features = {}
        # input_features["weekOfMonth"] = keras.Input(
        #     shape=(1,), name="weekOfMonth", dtype=tf.int64)
        # input_features["cityType"] = keras.Input(
        #     shape=(1,), name="cityType", dtype=tf.int64)
        input_features = {name: keras.Input(shape=(1,) if size == 1 else (size,),
                                            name=name, dtype=dtype)
                          for name, size, dtype, _ in self.feat_props if name != "label"}
        return input_features

    def create_embedding_dict(self):
        sparse_embedding = {}
        l2_reg = 1e-6
        for _, featName, _, featBucket, *_, featDefaultV, featDtype in self.cat_feat:
            print('featname{} bucket{}'.format(featName,featBucket))
            cur_emb = Embedding(featBucket, 10,  # 4 必须大于该特征的最大索引值
                                embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                embeddings_regularizer=l2(l2_reg),
                                name='sparse_emb_' + featName,
                                input_length=1)
            cur_emb.trainable = True
            sparse_embedding[featName] = cur_emb
        # weekOfMonth_emb = Embedding(4, 10,  # 4 必须大于该特征的最大索引值
        #                             embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
        #                             embeddings_regularizer=l2(l2_reg),
        #                             name='sparse_emb_weekOfMonth')
        # weekOfMonth_emb.trainable = True
        # sparse_embedding["weekOfMonth"] = weekOfMonth_emb
        #
        # cityType_emb = Embedding(10, 10,  # 必须大于该特征的最大索引值
        #                          embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
        #                          embeddings_regularizer=l2(l2_reg),
        #                          name='sparse_emb_cityType')
        # cityType_emb.trainable = True
        # sparse_embedding["cityType"] = cityType_emb

        return sparse_embedding

    def dnn_input_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)
        for _, featName, _, featBucket, *_, featDefaultV, featDtype in self.cat_feat:
            feat_lookup_idx = sparse_input_dict[featName]
            # group_embedding_dict['default_group'].append(
            #     tf.squeeze(sparse_embedding_dict[featName](feat_lookup_idx), axis=1))
            group_embedding_dict['default_group'].append(
                tf.keras.layers.Flatten()(sparse_embedding_dict[featName](feat_lookup_idx)))
        # user_lookup_idx = sparse_input_dict["weekOfMonth"]
        # group_embedding_dict['default_group'].append(sparse_embedding_dict["weekOfMonth"](user_lookup_idx))
        #
        # user_lookup_idx = sparse_input_dict["cityType"]
        # group_embedding_dict['default_group'].append(sparse_embedding_dict["cityType"](user_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def concat_func(self, inputs, axis=-1, mask=False):
        if not mask:
            # inputs = list(map(NoMask(), inputs))
            inputs
        if len(inputs) == 1:
            return inputs[0]
        else:
            return tf.keras.layers.Concatenate(axis=axis)(inputs)

    def fm_cross(self, embeddings):
        square_sum_tensor = tf.math.square(tf.math.reduce_sum(embeddings, axis=1))
        sum_square_tensor = tf.math.reduce_sum(tf.math.square(embeddings), axis=1)
        return 0.5 * tf.math.reduce_sum(square_sum_tensor - sum_square_tensor, axis=1, keepdims=True)

    def get_dense_input(self, features):
        dense_input_list = []
        for _, featName, _, featBucket, *_, featDefaultV, featDtype in self.num_feat:
            if featName == 'label':
                continue
            dense_input_list.append(features[featName])
        # dense_input_list.append(features["pay_score"])
        return dense_input_list

    def get_dense_input_v2(self, features):
        dense_input_list = []
        for _, featName, _, featBucket, *_, featDefaultV, featDtype in self.num_feat:
            if featName == 'label' or featName == 'dayDecayWeight':
                continue
            dense_input_list.append(features[featName])
        cosine = keras.layers.Dot(axes=1, normalize=True)(
            [features['userEmbedding'], features['titleEditedVectorBert']])
        dense_value_list = tf.keras.layers.Concatenate(axis=1)(dense_input_list)
        num_dense_feats = keras.layers.Concatenate()([dense_value_list, cosine])
        return num_dense_feats

    def build(self):
        input_features = self.buildInput()
        inputs_list = list(input_features.values())
        embDic = self.create_embedding_dict()
        # cat特征转embedding
        dnn_input_emb_list = self.dnn_input_embedding_lookup(embDic, input_features)
        input_dnn_layer = self.concat_func(dnn_input_emb_list)
        # squeeze：python3.6,tf2.1.0 下有问题: ValueError: Could not find matching function to call loaded from the SavedModel
        # 改为 python3.8，tf2.3.0就可以
        # input_dnn_layer = tf.compat.v2.squeeze(deep_input_emb,axis=1)

        # dense_value_list = self.get_dense_input(input_features)
        dense_value_list = self.get_dense_input_v2(input_features)
        # dense_value_list = tf.keras.layers.Concatenate(axis=1)(dense_value_list)
        lr_input_layer = tf.keras.layers.Concatenate(axis=1)([dense_value_list, input_dnn_layer])
        lr_input_layer = tf.keras.layers.Dense(units=1, name="LinearLayer")(lr_input_layer)

        # input_dnn_layer = tf.keras.layers.Flatten()(deep_input_emb)
        for hidden_unit in [32, 16]:
            input_dnn_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)(
                input_dnn_layer)
            input_dnn_layer = tf.keras.layers.Dropout(rate=0.2)(input_dnn_layer)
        dnn_logits_layer = tf.keras.layers.Dense(units=1, activation=None)(input_dnn_layer)

        fm_input_emb = tf.stack(dnn_input_emb_list, axis=1)
        square_sum_tensor = tf.math.square(tf.math.reduce_sum(fm_input_emb, axis=1))
        sum_square_tensor = tf.math.reduce_sum(tf.math.square(fm_input_emb), axis=1)
        fm_input_layer = 0.5 * tf.math.reduce_sum(square_sum_tensor - sum_square_tensor, axis=1, keepdims=True)
        # fm_input_layer = tf.keras.layers.Lambda(self.fm_cross, name="FmCrossLayer")(fm_input_emb)   使用lambda层会导致无法加载模型

        predict = tf.keras.layers.Add()(inputs=[dnn_logits_layer, lr_input_layer, fm_input_layer])
        predict = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                        kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(predict)
        model = tf.keras.models.Model(inputs=inputs_list, outputs=[predict])
        loss = keras.losses.binary_crossentropy

        return model, loss


if __name__ == "__main__":
    x = v14_5_8.data_gen(["F:\\data\\tensorflow\\v14_5_8\\date=20221115\\train\\part-r-00018",
                                "F:\\data\\tensorflow\\v14_5_8\\date=20221115\\train\\part-r-00019"], 4)
    v = v14_5_8.data_gen("F:\\data\\tensorflow\\v14_5_8\\date=20221115\\val\\part-r-00005", 4)
    t = v14_5_8.data_gen("F:\\data\\tensorflow\\v14_5_8\\date=20221115\\test\\part-r-00003", 4)

    test_op = tf.compat.v1.data.make_one_shot_iterator(x)
    one_element = test_op.get_next()
    print(one_element)
    print("one_element=>", one_element)
    # embModel = testEmb()
    # model, loss = embModel.build()
    # model.summary()
    # model.output_names[0] = 'predict_score'
    # optimizer = keras.optimizers.Adam()
    # metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(),
    #            keras.metrics.Recall(), keras.metrics.AUC()]
    # model.compile(loss=loss, optimizer=optimizer, metrics=metrics, experimental_run_tf_function=False)
    # history = model.fit(x,
    #                     validation_data=v,
    #                     epochs=2,
    #                     steps_per_epoch=1000)
    # model.save("v14_5_8_model")
    #
    # loadModel = keras.models.load_model("v14_5_8_model")
    # preRs = loadModel.predict(t)
    # print(preRs)
