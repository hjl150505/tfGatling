import tensorflow as tf
from tensorflow import keras, feature_column as fc
from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from collections import defaultdict
from itertools import chain
import numpy as np
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tfDataSet import din_fc_parser
from tensorflow.keras.experimental import SequenceFeatures

"""
python:3.6
tensorflow:2.3.0
mask=False
训练通过
保存模型通过
加载模型预测通过
"""


# tf.config.experimental_run_functions_eagerly(True)


class Dice(tf.keras.layers.Layer):

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        # tf.layers.batch_normalization(
        # inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerDin(keras.layers.Layer):
    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=False, blinding=True, seed=1024,
                 attention_type="scaled_dot_product", output_type="mean", use_bn=False, seq_len_max=10, **kwargs):

        assert head_num > 0, 'head_num must be a int > 0'
        self.hidden_units = []
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
        self.wq = tf.keras.layers.Dense(self.num_units)
        self.wk = tf.keras.layers.Dense(self.num_units)
        self.wv = tf.keras.layers.Dense(self.num_units)
        self.use_bn = use_bn
        self.seq_len_max = seq_len_max

        super(TransformerDin, self).__init__(**kwargs)

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        assert self.num_units == embedding_size, "embedding_size != num_units"
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.ln = tf.keras.layers.LayerNormalization()
        self.drop = tf.keras.layers.Dropout(rate=0.2)

        ########################################### din  begin ###########################################
        size = 4 * \
               int(input_shape[0][-1]
                   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.atten_kernel = self.add_weight(shape=(size, 1),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            name="kernel")
        self.atten_bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")

        hidden_units = [48, 80, 40]
        self.dnn_kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(
                                                hidden_units[i], hidden_units[i + 1]),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(len(self.hidden_units))]
        self.dnn_bias = [self.add_weight(name='bias' + str(i),
                                         shape=(self.hidden_units[i],),
                                         initializer=Zeros(),
                                         trainable=True) for i in range(len(self.hidden_units))]
        # if self.use_bn:
        if 0:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [Dice() for _ in range(len(self.hidden_units))]

        #################    dnn - end ##################################

        super(TransformerDin, self).build(input_shape)

    def positional_encoding(self, inputs,seq_max_len,
                            pos_embedding_trainable=True,
                            zero_pad=False,
                            scale=True,
                            ):
        _, _, num_units = inputs.get_shape().as_list()
        T = seq_max_len
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
            lookup_table = tf.constant(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5
        return outputs + inputs

    def call(self, inputs, mask=None, training=None, **kwargs):

        queries, keys, query_masks, key_masks, queryEmb = inputs
        keys_length_shape = key_masks.shape.as_list()
        key_emb_masks = key_masks
        if len(keys_length_shape) < 2:
            key_emb_masks = tf.expand_dims(key_emb_masks, axis=-1)
        key_emb_masks = tf.sequence_mask(
            key_emb_masks, self.seq_len_max)
        query_masks = tf.sequence_mask(
            query_masks, self.seq_len_max, dtype=tf.float32)
        key_masks = tf.sequence_mask(
            key_masks, self.seq_len_max, dtype=tf.float32)
        # query_masks = tf.squeeze(query_masks, axis=1)
        # key_masks = tf.squeeze(key_masks, axis=1)

        if self.use_positional_encoding:
            queries = self.positional_encoding(queries,self.seq_len_max)
            keys = self.positional_encoding(queries,self.seq_len_max)
        #
        querys = tf.tensordot(queries, self.W_Query,
                              axes=(-1, 0))  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))
        # tf.print(values.get_shape())   # [batchsize,seq_length,2*emb_size]

        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)
        # tf.print(values.get_shape())   # [batchsize,seq_length,2*emb_size/num_heads]

        if self.attention_type == "scaled_dot_product":
            # head_num*None T_q T_k
            outputs = tf.matmul(querys, keys, transpose_b=True)
            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        ############  ##########
        key_masks = tf.tile(key_masks, [self.head_num, 1])

        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

        # (h*N, T_q, T_k)

        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )
        # ############  ##########
        #
        outputs -= tf.reduce_max(outputs, axis=-1, keepdims=True)
        outputs = tf.nn.softmax(logits=outputs, axis=-1)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # # (h*N, T_q, T_k)
        query_masks = tf.tile(tf.expand_dims(
            query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks
        # #
        outputs = self.drop(outputs, training=training)
        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            result += queries
        if self.use_layer_norm:
            result = self.ln(result)

        if self.use_feed_forward:
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.drop(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)
        # result = tf.reduce_sum(result,
        #                        axis=1,
        #                        keepdims=True,
        #                        name="tran_sum")
        # result = tf.keras.layers.Flatten()(result)
        # tf.print(result.get_shape())  # [bachsize,2*embsize]
        ######################################### din begin ##########################################

        # key_masks = tf.expand_dims(mask[-1], axis=1)

        keys_len = result.get_shape()[1]
        queries = K.repeat_elements(queryEmb, keys_len, 1)
        # tf.print("*"*20)
        # tf.print(keys_len)
        # tf.print(result.get_shape())
        # tf.print(queries.get_shape())
        # tf.print(queries)
        # tf.print("*" * 20)

        att_input = tf.concat(
            [queries, result, queries - result, queries * result], axis=-1)

        att_out = att_input
        #
        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                att_out, self.dnn_kernels[i], axes=(-1, 0)), self.dnn_bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            att_out = fc

        attention_score = tf.nn.bias_add(tf.tensordot(
            att_out, self.atten_kernel, axes=(-1, 0)), self.atten_bias)

        # tf.print("-"*20)
        # tf.print(attention_score.get_shape())
        # tf.print("-" * 20)
        outputs = tf.transpose(attention_score, (0, 2, 1))

        paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_emb_masks, outputs, paddings)
        #
        # # if self.weight_normalization:
        # if 0:
        #     self.outputs = self.softmax(outputs)
        #
        # tf.print("+-" * 20)
        # tf.print(outputs.get_shape())
        # tf.print("+-" * 20)

        outputs = tf.matmul(outputs, result)
        # tf.print("+"*20)
        # tf.print(outputs.get_shape())
        # tf.print("+" * 20)
        ########################################## din end ###############################

        return outputs


class BST():
    def __init__(self):
        self.query_keys_featName_map = {}
        self.att_head_num = 4
        self.dnn_dropout = 0.2
        self.feat_props, self.num_feat, self.cat_feat, self.seq_feat = din_fc_parser.getFeatParserInfo()
        pass

    def build_num_inputs(self):
        num_inputs = {}
        num_cat_inputs = {}
        num_fcs = []
        num_cat_fcs = []
        num_fea_use_names = []
        num_fea_no_use_names = []

        def norm(mean_, stddev_):
            def scale(x):
                return (x - mean_) / stddev_

            return scale

        for use, name, size, mean, stddev, normalize, default, dtype, in self.num_feat:
            if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight':
                continue
            shape = (1,) if size == 1 else (size,)
            input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            feat_col = Lambda(norm(mean, stddev), name=name + "_lamb")(input_) if normalize else input_
            if use:
                num_fcs.append(feat_col)
                num_inputs[name] = input_
                num_fea_use_names.append(name)
            else:
                num_cat_fcs.append(feat_col)
                num_cat_inputs[name] = input_
                num_fea_no_use_names.append(name)

        cosine = keras.layers.Dot(axes=1, normalize=True)(
            [num_inputs['userEmbedding'], num_inputs['titleEditedVectorBert']])
        num_dense_feats = keras.layers.Concatenate()(num_fcs)
        num_dense_feats = keras.layers.Concatenate()([num_dense_feats, cosine])
        print("连续特征-使用=>", num_fea_use_names)
        print("连续特征-不用=>", num_fea_no_use_names)
        return num_dense_feats, num_inputs

    def build_cat_input_v2(self):
        cat_inputs = {}
        cat_fcs = []
        cat_fcs_ident = []
        cat_fea_names = []
        queryEmb = {}
        for use, name, size, buckets, weight_fc, default, dtype in self.cat_feat:
            if dtype != tf.string or name == 'userId':
                continue
            cat_fea_names.append(name)
            shape = (1,) if size == 1 else (1, size)
            input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            feat_col = fc.categorical_column_with_hash_bucket(key=name, hash_bucket_size=buckets)
            if size > 1:
                feat_col = fc.weighted_categorical_column(categorical_column=feat_col, weight_feature_key=weight_fc)
            dimension = 10
            cat_fc = fc.embedding_column(categorical_column=feat_col, dimension=dimension, combiner='sum')
            cat_fcs.append(keras.layers.DenseFeatures(cat_fc)({name: input_}))
            cat_inputs[name] = input_
            cat_fcs_ident.append(cat_fc)
            if name == "category" or name == "subCategory":
                queryEmb[name] = keras.layers.DenseFeatures(cat_fc)({name: input_})

        print("分类特征-使用=>", cat_fea_names)
        cat_inputs = {**cat_inputs}
        cat_dense_feats_emb = keras.layers.DenseFeatures(cat_fcs_ident)(cat_inputs)
        cat_dense_feats = tf.stack(cat_fcs, axis=1)

        return cat_dense_feats, cat_dense_feats_emb, cat_inputs, queryEmb

    def build_seq_input(self, keys2queryNameDic, catFeat):
        seq_keys_feat = {}
        seq_keys_feat_len = {}
        seq_inputs = {}
        cat_query_feat = {}
        seq_fea_len_dic = {}
        seq_feat_list = []
        for use, name, size, buckets, weight_fc, default, dtype in self.seq_feat:
            input_ = keras.layers.Input(name=name, shape=(None,), dtype=dtype)
            feat_col = fc.sequence_categorical_column_with_hash_bucket(key=name, hash_bucket_size=buckets, dtype=dtype)
            seq_emb = fc.embedding_column(categorical_column=feat_col, dimension=10)
            sequence_input = SequenceFeatures([seq_emb], trainable=True)
            seq_feat_emb, seq_fea_len = sequence_input({name: input_})
            seq_keys_feat[name] = seq_feat_emb
            # seq_feat_list.append(seq_feat_emb)
            seq_keys_feat_len[name] = seq_fea_len
            seq_inputs[name] = input_
            seq_fea_len_dic[name] = size
            cat_query_feat[name] = catFeat[keys2queryNameDic[name]]

        return seq_fea_len_dic, seq_inputs, seq_keys_feat, seq_keys_feat_len, cat_query_feat

    def build(self):
        hidden_units = [64, 32]
        dropout = 0.2
        self.query_keys_featName_map = {"feedClickCatList": "category", "feedClickSubCatList": "subCategory"}
        num_dense_feats, num_inputs = self.build_num_inputs()
        cat_dense_feats, cat_dense_feats_ident, cat_inputs, queryEmb = self.build_cat_input_v2()
        seq_fea_len, seq_inputs, seq_keys_feat, seq_keys_feat_len, cat_query_feat = self.build_seq_input(
            self.query_keys_featName_map, queryEmb)
        seq_mask_len = list(seq_keys_feat_len.values())[0]
        hist_emb = tf.keras.layers.Concatenate()(list(seq_keys_feat.values()))
        query_emb = tf.keras.layers.Concatenate()(list(queryEmb.values()))
        if query_emb.get_shape().ndims==2:
            query_emb = tf.expand_dims(query_emb,1)
        att_emb_size = hist_emb.get_shape().as_list()[-1] // self.att_head_num

        transformer_din_output = TransformerDin(att_embedding_size=att_emb_size, head_num=self.att_head_num,
                                                dropout_rate=self.dnn_dropout, use_positional_encoding=True,
                                                use_res=True,
                                                use_feed_forward=True, use_layer_norm=True, blinding=False, seed=2022)(
            [hist_emb, hist_emb, seq_mask_len, seq_mask_len, query_emb])

        input_dnn_layer = transformer_din_output
        for hidden_unit in hidden_units:
            input_dnn_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)(
                input_dnn_layer)
            input_dnn_layer = tf.keras.layers.Dropout(rate=dropout)(input_dnn_layer)

        dnn_logits_layer = tf.keras.layers.Dense(units=1, activation=None)(input_dnn_layer)
        predict = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                        kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(dnn_logits_layer)
        model = tf.keras.models.Model(inputs={**cat_inputs, **seq_inputs}, outputs=[predict])
        # model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[predict])
        loss = keras.losses.binary_crossentropy

        return model, loss


if __name__ == "__main__":
    x = din_fc_parser.data_gen(["F:\\data\\tensorflow\\v23_2_1\\date=20221120\\train\\part-r-00000",
                                "F:\\data\\tensorflow\\v23_2_1\\date=20221120\\train\\part-r-00001"], 4)
    v = din_fc_parser.data_gen("F:\\data\\tensorflow\\v23_2_1\\date=20221120\\val\\part-r-00002", 4)
    t = din_fc_parser.data_gen("F:\\data\\tensorflow\\v23_2_1\\date=20221120\\test\\part-r-00003", 4)
    test_op = tf.compat.v1.data.make_one_shot_iterator(t)
    one_element = test_op.get_next()
    print(one_element)
    print("one_element=>", one_element)
    bst = BST()
    model,loss = bst.build()
    model.output_names[0] = 'predict_score'
    optimizer = keras.optimizers.Adam()
    metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(),
               keras.metrics.Recall(), keras.metrics.AUC()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, experimental_run_tf_function=False)
    model.summary()
    # predictRs_1 = model.predict(v)
    # print(predictRs_1)
    print("+"*20)
    history = model.fit(x,
                        validation_data=v,
                        epochs=2,
                        steps_per_epoch=1000)
    model.save("bst_model_rs_fc")

    loadModel = keras.models.load_model("bst_model_rs_fc")
    print(loadModel.predict(t))
