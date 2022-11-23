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
from tensorflow.python.keras.layers import Lambda

from tfDataSet import v23_1_1
from dinUnit_v23_1_1 import *
from parseFeatSchema import v23_1_1parserTsv

"""
测试embedding层的使用
python=3.6
tensorflow=2.3.0
训练成功
模型加载、预测成功
注意：模型输入的是id型特征，因此必须保证所有数据集的id分桶必须一致，否则报错：tensorflow.python.framework.errors_impl.InvalidArgumentError:  Incompatible shapes at component 92: expected [?] but got [].
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

    def query_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)
        query_names = ['category', 'subCategory']
        # query_names = ['item_id', 'cat_id']
        for name in query_names:
            id_lookup_idx = sparse_input_dict[name]
            group_embedding_dict['default_group'].append(sparse_embedding_dict[name](id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def keys_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)
        keys_names = ['category', 'subCategory']

        for name in keys_names:
            id_lookup_idx = sparse_input_dict[name + '_his']
            group_embedding_dict['default_group'].append(sparse_embedding_dict[name](id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def dnn_input_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)
        for use, name, size, buckets, weight_fc, default, dtype in self.cat_feat:
            if name in ['userId', 'category_his', 'subCategory_his']:
                continue
            id_lookup_idx = sparse_input_dict[name]
            group_embedding_dict['default_group'].append(tf.keras.layers.Flatten()(sparse_embedding_dict[name](id_lookup_idx)))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def get_dense_input(self, features):
        def norm(mean_, stddev_):
            def scale(x):
                return (x - mean_) / stddev_

            return scale
        dense_input_list = []
        for use, name, size, mean, stddev, normalize, default, dtype, in self.num_feat:
            if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight':
                continue
            # dense_input_list.append(features[name])
            feat_col = Lambda(norm(mean, stddev), name=name + "_lamb")(features[name]) if normalize else features[name]
            dense_input_list.append(feat_col)
        cosine = keras.layers.Dot(axes=1, normalize=True)(
            [features['userEmbedding'], features['titleEditedVectorBert']])
        dense_input_list.append(cosine)
        return dense_input_list

    def concat_func(self, inputs, axis=-1, mask=False):
        if not mask:
            # inputs = list(map(NoMask(), inputs))
            inputs
        if len(inputs) == 1:
            return inputs[0]
        else:
            return tf.keras.layers.Concatenate(axis=axis)(inputs)

    def combined_dnn_input(self, sparse_embedding_list, dense_value_list):
        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            sparse_dnn_input = keras.layers.Flatten()(self.concat_func(sparse_embedding_list))
            dense_dnn_input = keras.layers.Flatten()(self.concat_func(dense_value_list))
            return self.concat_func([sparse_dnn_input, dense_dnn_input])
        elif len(sparse_embedding_list) > 0:
            return keras.layers.Flatten()(self.concat_func(sparse_embedding_list))
        elif len(dense_value_list) > 0:
            return keras.layers.Flatten()(self.concat_func(dense_value_list))
        else:
            raise NotImplementedError("dnn_feature_columns can not be empty list")

    def build(self):
        hidden_units = [32, 16]
        dropout = 0.2
        input_features, embDic = self.buildInput()
        inputs_list = list(input_features.values())
        query_emb_list = self.query_embedding_lookup(embDic, input_features)
        keys_emb_list = self.keys_embedding_lookup(embDic, input_features)
        dnn_input_emb_list = self.dnn_input_embedding_lookup(embDic, input_features)
        dense_value_list = self.get_dense_input(input_features)

        query_emb = self.concat_func(query_emb_list, mask=True)
        keys_emb = self.concat_func(keys_emb_list, mask=True)
        deep_input_emb = self.concat_func(dnn_input_emb_list)
        fm_input_emb = tf.stack(dnn_input_emb_list,axis=1)
        square_sum_tensor = tf.math.square(tf.math.reduce_sum(fm_input_emb, axis=1))
        sum_square_tensor = tf.math.reduce_sum(tf.math.square(fm_input_emb), axis=1)
        fm_input_layer = 0.5 * tf.math.reduce_sum(square_sum_tensor - sum_square_tensor, axis=1, keepdims=True)

        hist = dinUnit((32, 16), 'dice', weight_normalization=False, supports_masking=True)([
            keys_emb, query_emb])
        hist = tf.keras.layers.Flatten()(hist)
        print(f'deep_input_emb_shape:{deep_input_emb.get_shape()}')
        print(f'hist_shape:{hist.get_shape()}')

        # deep_input_emb = tf.concat([deep_input_emb, hist], axis=-1)
        lr_input_layer = self.combined_dnn_input([deep_input_emb], dense_value_list)
        lr_input_layer = tf.keras.layers.Dense(units=1, name="LinearLayer")(lr_input_layer)
        dnn_input = tf.keras.layers.Concatenate()(dnn_input_emb_list+[hist])
        for hidden_unit in hidden_units:
            dnn_input = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)(
                dnn_input)
            dnn_input = tf.keras.layers.Dropout(rate=dropout)(dnn_input)
        dnn_logits_layer = tf.keras.layers.Dense(units=1, activation=None)(dnn_input)

        predict = tf.keras.layers.Add()(inputs=[lr_input_layer, fm_input_layer, dnn_logits_layer])

        predict = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                        kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(predict)

        model = tf.keras.models.Model(inputs=inputs_list, outputs=[predict])

        loss = keras.losses.binary_crossentropy
        return model, loss


if __name__ == "__main__":
    x = v23_1_1.data_gen(["F:\\data\\tensorflow\\v23_1_1\\date=20221117\\train\\part-r-00000",
                          "F:\\data\\tensorflow\\v23_1_1\\date=20221117\\train\\part-r-00001"], 4)
    v = v23_1_1.data_gen("F:\\data\\tensorflow\\v23_1_1\\date=20221117\\val\\part-r-00000", 4)
    t = v23_1_1.data_gen("F:\\data\\tensorflow\\v23_1_1\\date=20221117\\test\\part-r-00000", 4)

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
