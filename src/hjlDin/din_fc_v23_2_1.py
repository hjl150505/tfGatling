import tensorflow as tf
from tensorflow import keras, feature_column as fc
from tensorflow.python.keras.regularizers import l1_l2
# from tensorflow.python.feature_column.sequence_feature_column import SequenceFeatures
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tfDataSet import din_fc_parser
from tensorflow.python.keras.layers import Lambda

"""
python3.6
TensorFlow：2.3.0
惊天大胜利：模型中可以添加归一化操作了
训练通过
加载模型通过
预测通过
"""


class DinFc():
    def __init__(self):
        self.query_keys_featName_map = {}
        self.feat_props, self.num_feat, self.cat_feat, self.seq_feat = din_fc_parser.getFeatParserInfo()

    def fm_cross(self, embeddings):
        square_sum_tensor = tf.math.square(tf.math.reduce_sum(embeddings, axis=1))
        sum_square_tensor = tf.math.reduce_sum(tf.math.square(embeddings), axis=1)
        return 0.5 * tf.math.reduce_sum(square_sum_tensor - sum_square_tensor, axis=1, keepdims=True)

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
            # normalizer_fn = norm(mean, stddev) if normalize else None

            # feat_col = fc.numeric_column(key=name, shape=shape, normalizer_fn=None)
            feat_col = Lambda(norm(mean, stddev), name=name + "_lamb")(input_) if normalize else input_
            if use:
                num_fcs.append(feat_col)
                num_inputs[name] = input_
                num_fea_use_names.append(name)
            else:
                num_cat_fcs.append(feat_col)
                num_cat_inputs[name] = input_
                num_fea_no_use_names.append(name)

        # user_emb_fc = [x for x in num_fcs if x.name.split('/')[0] == 'userEmbedding'][0]
        # user_emb_in = {user_emb_fc.name.split('/')[0]: num_inputs[user_emb_fc.name.split('/')[0]]}
        # user_emb = keras.layers.DenseFeatures(user_emb_fc)(user_emb_in)
        #
        # item_emb_fc = [x for x in num_fcs if x.name.split('/')[0] == 'titleEditedVectorBert'][0]
        # item_emb_in = {item_emb_fc.name: num_inputs[item_emb_fc.name.split('/')[0]]}
        # item_emb = keras.layers.DenseFeatures(item_emb_fc)(item_emb_in)

        cosine = keras.layers.Dot(axes=1, normalize=True)(
            [num_inputs['userEmbedding'], num_inputs['titleEditedVectorBert']])
        # num_dense_feats = keras.layers.DenseFeatures(num_fcs)(num_inputs)
        # num_dense_feats = keras.layers.Concatenate()([num_dense_feats, cosine])
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
        for use, name, size, buckets, weight_fc, default, dtype in self.seq_feat:
            input_ = keras.layers.Input(name=name, shape=(size,), dtype=dtype)
            feat_col = fc.sequence_categorical_column_with_hash_bucket(key=name, hash_bucket_size=buckets, dtype=dtype)
            seq_emb = fc.embedding_column(categorical_column=feat_col, dimension=10)
            sequence_input = SequenceFeatures([seq_emb], trainable=True)
            seq_feat_emb, seq_fea_len = sequence_input({name: input_})
            seq_keys_feat[name] = seq_feat_emb
            seq_keys_feat_len[name] = seq_fea_len
            seq_inputs[name] = input_
            seq_fea_len_dic[name] = size
            cat_query_feat[name] = catFeat[keys2queryNameDic[name]]

        return seq_fea_len_dic, seq_inputs, seq_keys_feat, seq_keys_feat_len, cat_query_feat

    def din_attention(self, keys, query, keys_length, hidden_layers, scale=False):
        """
        :param keys: [B, T, H]
        :param query: [B, H]
        :param keys_length: [B, 1]
        :return: attention_weighted embedding: [B, H]
        """
        query_shape = query.shape.as_list()
        keys_shape = keys.shape.as_list()
        keys_length_shape = keys_length.shape.as_list()
        if len(keys_length_shape) < 2:
            keys_length = tf.expand_dims(keys_length, axis=-1)
        print("keys shape", keys_shape, ", query shape", query_shape, ", keys_length", keys_length.shape)
        assert query_shape[-1] == keys_shape[-1], "keys embedding size should be equal to query embedding size"
        assert len(keys_shape) == 3, "keys bad shape"

        query = tf.expand_dims(query, -2)  # B*1*H
        T = tf.shape(keys)[1]
        query = tf.tile(query, [1, T, 1])  # B*T*H

        concat_layer = tf.concat([query, keys, query - keys, query * keys], axis=-1)  # B*T*(H+H+H+H)
        mlp_out = concat_layer
        activation_f = tf.keras.activations.sigmoid  # tf.nn.leaky_relu
        for hidden_unit in hidden_layers:
            mlp_out = tf.keras.layers.Dense(units=hidden_unit,
                                            kernel_initializer=tf.keras.initializers.he_uniform(999),
                                            activation=activation_f)(mlp_out)
            mlp_out = keras.layers.BatchNormalization()(mlp_out)
        mlp_out = tf.keras.layers.Dense(units=1, activation=activation_f)(mlp_out)  # B*T*1
        weights = tf.transpose(mlp_out, perm=[0, 2, 1])  # B*1*T
        mask_max_len = tf.shape(keys)[1]
        key_masks = tf.sequence_mask(keys_length, mask_max_len)  # B*1*T

        if scale:  # B*1*T
            paddings = tf.ones_like(weights) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(weights)
        weights = tf.where(key_masks, weights, paddings)
        if scale:
            weights = weights / (keys_shape[-1] ** 0.5)
            weights = tf.nn.softmax(weights)
        weighted_res = tf.matmul(weights, keys)  # B*1*H
        weighted_res = tf.keras.layers.Flatten()(weighted_res)  # B*H
        return weighted_res

    def build(self):
        hidden_units = [64, 32]
        dropout = 0.2
        self.query_keys_featName_map = {"feedClickCatList": "category", "feedClickSubCatList": "subCategory"}
        num_dense_feats, num_inputs = self.build_num_inputs()
        cat_dense_feats, cat_dense_feats_ident, cat_inputs, queryEmb = self.build_cat_input_v2()

        lr_input_layer = tf.keras.layers.Concatenate(axis=1)([num_dense_feats, cat_dense_feats_ident])
        lr_input_layer = tf.keras.layers.Dense(units=1, name="LinearLayer")(lr_input_layer)

        # fm_input_layer = tf.keras.layers.Lambda(self.fm_cross, name="FmCrossLayer")(cat_dense_feats)
        square_sum_tensor = tf.math.square(tf.math.reduce_sum(cat_dense_feats, axis=1))
        sum_square_tensor = tf.math.reduce_sum(tf.math.square(cat_dense_feats), axis=1)
        fm_input_layer = 0.5 * tf.math.reduce_sum(square_sum_tensor - sum_square_tensor, axis=1, keepdims=True)

        # din test
        seq_fea_len, seq_inputs, seq_keys_feat, seq_keys_feat_len, cat_query_feat = self.build_seq_input(
            self.query_keys_featName_map, queryEmb)
        attention_outs = []
        for attention_key in seq_keys_feat:
            attention_o = self.din_attention(seq_keys_feat[attention_key],
                                             cat_query_feat[attention_key],
                                             seq_keys_feat_len[attention_key],
                                             [32, 16],
                                             False)
            attention_o_norm = keras.layers.BatchNormalization()(attention_o)
            attention_outs.append(attention_o_norm)
        print("attention out shape:", [i.shape for i in attention_outs])
        attention_out_concat = tf.keras.layers.Concatenate(axis=1)(attention_outs)
        dnn_dense_feats = tf.keras.layers.Concatenate(axis=1)([cat_dense_feats_ident, attention_out_concat])
        input_dnn_layer = tf.keras.layers.Flatten()(dnn_dense_feats)
        for hidden_unit in hidden_units:
            input_dnn_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)(
                input_dnn_layer)
            input_dnn_layer = tf.keras.layers.Dropout(rate=dropout)(input_dnn_layer)
        dnn_logits_layer = tf.keras.layers.Dense(units=1, activation=None)(input_dnn_layer)

        predict = tf.keras.layers.Add()(inputs=[lr_input_layer, fm_input_layer, dnn_logits_layer])

        predict = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                        kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(predict)
        model = tf.keras.models.Model(inputs={**num_inputs, **cat_inputs, **seq_inputs}, outputs=[predict])
        # model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[predict])
        loss = keras.losses.binary_crossentropy

        return model, loss
        pass


def trainMain():
    x = din_fc_parser.data_gen(["F:\\data\\tensorflow\\v23_2_1\\date=20221120\\train\\part-r-00000",
                                "F:\\data\\tensorflow\\v23_2_1\\date=20221120\\train\\part-r-00001"], 4)
    v = din_fc_parser.data_gen("F:\\data\\tensorflow\\v23_2_1\\date=20221120\\val\\part-r-00002", 4)
    t = din_fc_parser.data_gen("F:\\data\\tensorflow\\v23_2_1\\date=20221120\\test\\part-r-00003", 4)
    test_op = tf.compat.v1.data.make_one_shot_iterator(t)
    one_element = test_op.get_next()
    print(one_element)
    print("one_element=>", one_element)
    embModel = DinFc()
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
    model.save("din_fc_model")
    loadModel = keras.models.load_model("din_fc_model")
    # 测试线上训练的模型（测试成功，可以预测）：
    # loadModel = keras.models.load_model("F:\\data\\tensorflow\\v23_2_1\\modelOnline\\saved_model\\br\hjlModels\\model_v23_2_1\\20221117")
    preRs = loadModel.predict(t)
    print(preRs)


def checkFcData():
    num_inputs = {}

    def norm(mean_, stddev_):
        def scale(x):
            return (x - mean_) / stddev_

        return scale

    t = din_fc_parser.data_gen("F:\\data\\tensorflow\\din_fc\\date=20220718\\test\\part-r-00000", 4)
    # test_op = tf.compat.v1.data.make_one_shot_iterator(t)
    # one_element = test_op.get_next()
    # print("one_element=>", one_element)
    *_, seq_feat = din_fc_parser.getFeatParserInfo()
    for use, name, size, bucket, normalize, default, dtype, in seq_feat:
        input_ = keras.layers.Input(name=name, shape=(None,), dtype=dtype)
        print("name=>" + name)
        num_inputs[name] = input_
        feat_col = fc.sequence_categorical_column_with_hash_bucket(key=name, hash_bucket_size=bucket, dtype=dtype)
        seq_emb = fc.embedding_column(categorical_column=feat_col, dimension=10)
        sequence_input = SequenceFeatures([seq_emb], trainable=True)
        seq_feat_emb, seq_fea_len = sequence_input({name: input_})

        # num_dense_feats = keras.layers.DenseFeatures(feat_col)(num_inputs)
        model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[seq_feat_emb])
        print("name=>" + name)
        preVelue = model.predict(t)
        preValueDecode = [preVelue[i][0].decode(encoding='UTF-8', errors='strict') for i in range(len(preVelue[0][0]))]
        # preValueDecode = preVelue.decode(encoding='UTF-8', errors='strict')
        print(preValueDecode)
        pass


if __name__ == "__main__":
    # checkFcData()
    trainMain()
    pass
