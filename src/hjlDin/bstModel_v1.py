import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from collections import defaultdict
from itertools import chain
import numpy as np
from tensorflow.python.keras import backend as K
from dinUnit import dinUnit

"""
python:3.6
tensorflow:2.1.0
可训练
不可保存模型
"""
# tf.config.experimental_run_functions_eagerly(True)

def positional_encoding(inputs,
                        pos_embedding_trainable=True,
                        zero_pad=False,
                        scale=True,
                        ):
    _, T, num_units = inputs.get_shape().as_list()
    # with tf.variable_scope(scope, reuse=reuse):
    position_ind = tf.expand_dims(tf.range(T), 0)
    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, 2. * i / num_units)
         for i in range(num_units)]
        for pos in range(T)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    # Convert to a tensor

    if pos_embedding_trainable:
        # lookup_table = K.variable(position_enc, dtype=tf.float32)
        # lookup_table = tf.Variable(position_enc, dtype=tf.float32)
        lookup_table = tf.constant(position_enc,dtype=tf.float32)

    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                  lookup_table[1:, :]), 0)

    outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

    if scale:
        outputs = outputs * num_units ** 0.5
    return outputs + inputs

class Transformer(keras.layers.Layer):
    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=False, blinding=True, seed=1024, supports_masking=False,
                 attention_type="scaled_dot_product", output_type="mean", **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        super(Transformer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def positional_encoding(self, inputs,
                            pos_embedding_trainable=True,
                            zero_pad=False,
                            scale=True,
                            ):
        _, T, num_units = inputs.get_shape().as_list()
        # with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.expand_dims(tf.range(T), 0)
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units)
             for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor

        if pos_embedding_trainable:
            # lookup_table = K.variable(position_enc, dtype=tf.float32)
            lookup_table = tf.Variable(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5
        return outputs + inputs

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        assert self.num_units == embedding_size, "embedding_size != num_units"
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))

        if self.attention_type == "additive":
            self.b = self.add_weight('b', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.v = self.add_weight('v', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        # if self.use_res:
        #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
        #                                  initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.ln = tf.keras.layers.LayerNormalization()

        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs

            query_masks = tf.sequence_mask(
                query_masks, self.seq_len_max, dtype=tf.float32)
            key_masks = tf.sequence_mask(
                key_masks, self.seq_len_max, dtype=tf.float32)
            # query_masks = tf.squeeze(query_masks, axis=1)
            # key_masks = tf.squeeze(key_masks, axis=1)

        if self.use_positional_encoding:
            # queries = self.positional_encoding(queries)
            # keys = self.positional_encoding(queries)
            queries = positional_encoding(queries)
            keys = positional_encoding(queries)

        querys = tf.tensordot(queries, self.W_Query,
                              axes=(-1, 0))  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))

        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":
            # head_num*None T_q T_k
            outputs = tf.matmul(querys, keys, transpose_b=True)

            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        # outputs -= reduce_max(outputs, axis=-1, keep_dims=True)
        outputs -= tf.reduce_max(outputs, axis=1, keepdims=True)
        # outputs = softmax(outputs)
        outputs = tf.nn.softmax(logits=outputs,axis=-1)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # (h*N, T_q, T_k)
        query_masks = tf.tile(tf.expand_dims(
            query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = tf.keras.layers.Dropout(rate=0.2)(outputs, training=training)
        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
            result += queries
        if self.use_layer_norm:
            # result = tf.keras.layers.LayerNormalization()(result)
            result = self.ln(result)

        if self.use_feed_forward:
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = tf.keras.layers.Dropout(rate=0.2)(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                # result = tf.keras.layers.LayerNormalization()(result)
                result = self.ln(result)
        return result


class BST:
    def __init__(self):
        pass

    def buildInput(self):
        input_features = {}
        input_features["user"] = keras.Input(
            shape=(1,), name="user", dtype=tf.int32)
        input_features["gender"] = keras.Input(
            shape=(1,), name="gender", dtype=tf.int32)
        input_features["item_id"] = keras.Input(
            shape=(1,), name="item_id", dtype=tf.int32)
        input_features["cate_id"] = keras.Input(
            shape=(1,), name="cate_id", dtype=tf.int32)
        input_features["pay_score"] = keras.Input(
            shape=(1,), name="pay_score", dtype=tf.float32)
        input_features["his_item_id"] = keras.Input(
            shape=(4,), name="his_item_id", dtype=tf.int32)
        input_features["seq_length"] = keras.Input(
            shape=(1,), name="seq_length", dtype=tf.int32)
        input_features["his_cate_id"] = keras.Input(
            shape=(4,), name="his_cate_id", dtype=tf.int32)
        return input_features

    def create_embedding_dict(self):
        sparse_embedding = {}
        l2_reg = 1e-6
        user_emb = Embedding(3, 10,
                             embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                             embeddings_regularizer=l2(l2_reg),
                             name='sparse_emb_user')
        user_emb.trainable = True
        sparse_embedding["user"] = user_emb

        gender_emb = Embedding(2, 4,
                               embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                               embeddings_regularizer=l2(l2_reg),
                               name='sparse_emb_gender')
        gender_emb.trainable = True
        sparse_embedding["gender"] = gender_emb

        item_id_emb = Embedding(4, 8,
                                embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                embeddings_regularizer=l2(l2_reg),
                                name='sparse_emb_item_id')
        item_id_emb.trainable = True
        sparse_embedding["item_id"] = item_id_emb

        cat_id_emb = Embedding(3, 4,
                               embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                               embeddings_regularizer=l2(l2_reg),
                               name='sparse_emb_cate_id')
        cat_id_emb.trainable = True
        sparse_embedding["cate_id"] = cat_id_emb

        his_item_id_emb = Embedding(4, 8,
                                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                    embeddings_regularizer=l2(
                                        l2_reg),
                                    name='sparse_seq_emb_hist_item_id',
                                    mask_zero=True)
        his_item_id_emb.trainable = True
        sparse_embedding["item_id"] = his_item_id_emb

        his_cat_id_emb = Embedding(3, 4,
                                   embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                   embeddings_regularizer=l2(
                                       l2_reg),
                                   name='sparse_seq_emb_hist_cate_id',
                                   mask_zero=True)
        his_cat_id_emb.trainable = True
        sparse_embedding["cate_id"] = his_cat_id_emb

        return sparse_embedding

    def query_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)

        item_id_lookup_idx = sparse_input_dict["item_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["item_id"](item_id_lookup_idx))
        cat_id_lookup_idx = sparse_input_dict["cate_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["cate_id"](cat_id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def keys_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)

        his_item_id_lookup_idx = sparse_input_dict["his_item_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["item_id"](his_item_id_lookup_idx))
        his_cat_id_lookup_idx = sparse_input_dict["his_cate_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["cate_id"](his_cat_id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def dnn_input_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)

        user_lookup_idx = sparse_input_dict["user"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["user"](user_lookup_idx))
        gender_lookup_idx = sparse_input_dict["gender"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["gender"](gender_lookup_idx))
        item_id_lookup_idx = sparse_input_dict["item_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["item_id"](item_id_lookup_idx))
        cat_id_lookup_idx = sparse_input_dict["cate_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["cate_id"](cat_id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def get_dense_input(self, features):
        dense_input_list = []
        dense_input_list.append(features["pay_score"])
        return dense_input_list

    def concat_func(self, inputs, axis=-1, mask=False):
        if not mask:
            # inputs = list(map(NoMask(), inputs))
            inputs
        if len(inputs) == 1:
            return inputs[0]
        else:
            return tf.keras.layers.Concatenate(axis=axis)(inputs)

    def build(self):
        transformer_num = 1
        att_head_num = 4
        dnn_dropout = 0.2
        seed = 2022
        input_features = self.buildInput()
        inputs_list = list(input_features.values())
        user_behavior_length = input_features["seq_length"]
        embDic = self.create_embedding_dict()
        query_emb_list = self.query_embedding_lookup(embDic, input_features)
        hist_emb_list = self.keys_embedding_lookup(embDic, input_features)
        dnn_input_emb_list = self.dnn_input_embedding_lookup(embDic, input_features)
        dense_value_list = self.get_dense_input(input_features)

        hist_emb = self.concat_func(hist_emb_list, mask=True)
        deep_input_emb = self.concat_func(dnn_input_emb_list)
        query_emb = self.concat_func(query_emb_list, mask=True)
        transformer_output = hist_emb
        # for i in range(transformer_num):
        #     att_embedding_size = transformer_output.get_shape().as_list()[-1] // att_head_num
        #     transformer_layer = Transformer(att_embedding_size=att_embedding_size, head_num=att_head_num,
        #                                     dropout_rate=dnn_dropout, use_positional_encoding=True, use_res=True,
        #                                     use_feed_forward=True, use_layer_norm=True, blinding=False, seed=seed,
        #                                     supports_masking=True, output_type=None)
            # transformer_output = transformer_layer([transformer_output, transformer_output,
            #                                         user_behavior_length, user_behavior_length])
            # transformer_output = transformer_layer([query_emb, transformer_output])

        hist = dinUnit((80, 40), 'dice', weight_normalization=False, supports_masking=True)([
            query_emb, transformer_output])
        deep_input_emb = tf.keras.layers.Concatenate()([deep_input_emb, hist])
        # deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
        final_logit_tmp = tf.keras.layers.Dense(1, use_bias=False)(deep_input_emb)
        model = tf.keras.models.Model(inputs=inputs_list, outputs=final_logit_tmp)
        model.summary()
        return model

def get_xy_fd():
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])
    seq_length = np.array([3, 3, 2])  # the actual length of the behavior sequence

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'his_item_id': hist_iid, 'his_cate_id': hist_cate_id,
                    'pay_score': pay_score, 'seq_length': seq_length}
    x = {name: feature_dict[name] for name in feature_dict.keys()}
    y = np.array([1, 0, 1])
    z = {'input_1': uid, 'input_2': ugender, 'input_3': iid, 'input_4': cate_id,
         'input_5': hist_iid, 'input_6': hist_cate_id,
         'input_7': pay_score, 'input_8': seq_length}
    return x, y, z


if __name__ == "__main__":
    x, y, z = get_xy_fd()
    bst = BST()
    model = bst.build()
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.run_eagerly = True
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)
    model.save("bst_model_rs")

    loadModel = keras.models.load_model("bst_model_rs")
    print(loadModel.predict(x))
