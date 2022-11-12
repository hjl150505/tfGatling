import tensorflow as tf
from tensorflow import keras, feature_column as fc
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2
from collections import defaultdict
from itertools import chain
from dinUnit import dinUnit
from DNN import DNN
from PredictionLayer import PredictionLayer
from lib.model.model_builder_base import ModelBuilderBase

"""
20221019：执行 summary 通过
"""
class AlchemyBuilder(ModelBuilderBase):
    def build(self):
        num_dense_feats, num_inputs, num_cat_fcs, num_cat_inputs = self.build_num_inputs()
        cat_dense_feats, cat_inputs = self.build_cat_inputs(num_cat_fcs=num_cat_fcs, num_cat_inputs=num_cat_inputs)

        # dnn
        in_ = keras.layers.Concatenate(axis=1)([num_dense_feats, cat_dense_feats])

        wide = in_
        deep = in_

        deep = keras.layers.Dropout(rate=0.2)(deep)
        deep = keras.layers.Dense(128, activation=None, kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(deep)
        deep = keras.layers.ReLU()(deep)

        deep = keras.layers.BatchNormalization()(deep)
        deep = keras.layers.Dropout(rate=0.2)(deep)
        deep = keras.layers.Dense(64, activation=None, kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(deep)
        deep = keras.layers.ReLU()(deep)

        deep = keras.layers.BatchNormalization()(deep)
        deep = keras.layers.Dropout(rate=0.2)(deep)
        deep = keras.layers.Dense(32, activation=None, kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(deep)
        deep = keras.layers.ReLU()(deep)

        out = keras.layers.Concatenate()([wide, deep])
        out = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(out)

        model = keras.Model(inputs={**num_inputs, **cat_inputs}, outputs=out)
        loss = keras.losses.binary_crossentropy

        return model, loss

    def build_cat_inputs(self, num_cat_fcs, num_cat_inputs):
        cat_inputs = {}
        cat_fcs = []
        cat_fea_names = []
        for use, name, size, buckets, weight_fc, default, dtype in self._cat_feat_props:
            if dtype != tf.string:
                continue
            cat_fea_names.append(name)
            shape = (1,) if size == 1 else (1, size)
            input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            feat_col = fc.categorical_column_with_hash_bucket(key=name, hash_bucket_size=buckets)

            if size > 1:
                feat_col = fc.weighted_categorical_column(categorical_column=feat_col, weight_feature_key=weight_fc)

            # dimension = math.floor(pow(buckets, 1 / 4)) + 1
            dimension = 1
            cat_fc = fc.embedding_column(categorical_column=feat_col,
                                         dimension=dimension,
                                         combiner='sum')

            cat_fcs.append(cat_fc)
            cat_inputs[name] = input_

        # add weight for sequence categorical feature
        cat_fcs = cat_fcs
        cat_inputs = {**cat_inputs, **num_cat_inputs}
        cat_dense_feats = keras.layers.DenseFeatures(cat_fcs)(cat_inputs)

        print("类别特征名：", cat_fea_names)

        return cat_dense_feats, cat_inputs

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

        for use, name, size, mean, stddev, normalize, default, dtype in self._num_feat_props:
            if (dtype != tf.float32) or (name == 'label'):
                continue

            shape = (1,) if size == 1 else (1, size)
            input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            normalizer_fn = norm(mean, stddev) if normalize else None
            feat_col = fc.numeric_column(key=name, shape=shape, normalizer_fn=normalizer_fn)

            if use:
                num_fcs.append(feat_col)
                num_inputs[name] = input_
                num_fea_use_names.append(name)
            else:
                num_cat_fcs.append(feat_col)
                num_cat_inputs[name] = input_
                num_fea_no_use_names.append(name)

        # only used feature is flowed into dense features
        # add cosine similarity between user embedding and item embedding
        user_emb_fc = [x for x in num_fcs if x.name == 'userEmbedding'][0]
        user_emb_in = {user_emb_fc.name: num_inputs[user_emb_fc.name]}
        user_emb = keras.layers.DenseFeatures(user_emb_fc)(user_emb_in)

        item_emb_fc = [x for x in num_fcs if x.name == 'titleEditedVectorBert'][0]
        item_emb_in = {item_emb_fc.name: num_inputs[item_emb_fc.name]}
        item_emb = keras.layers.DenseFeatures(item_emb_fc)(item_emb_in)

        cosine = keras.layers.Dot(axes=1, normalize=True)([user_emb, item_emb])

        num_dense_feats = keras.layers.DenseFeatures(num_fcs)(num_inputs)
        num_dense_feats = keras.layers.Concatenate()([num_dense_feats, cosine])

        print("连续特征_use：", num_fea_use_names)
        print("连续特征_no_use：", num_fea_no_use_names)

        return num_dense_feats, num_inputs, num_cat_fcs, num_cat_inputs


class FmBuider(ModelBuilderBase):
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

        for use, name, size, mean, stddev, normalize, default, dtype, in self._num_feat_props:
            if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight':
                continue
            shape = (1,) if size == 1 else (1, size)
            input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            normalizer_fn = norm(mean, stddev) if normalize else None
            feat_col = fc.numeric_column(key=name, shape=shape, normalizer_fn=normalizer_fn)

            if use:
                num_fcs.append(feat_col)
                num_inputs[name] = input_
                num_fea_use_names.append(name)
            else:
                num_cat_fcs.append(feat_col)
                num_cat_inputs[name] = input_
                num_fea_no_use_names.append(name)

        user_emb_fc = [x for x in num_fcs if x.name == 'userEmbedding'][0]
        user_emb_in = {user_emb_fc.name: num_inputs[user_emb_fc.name]}
        user_emb = keras.layers.DenseFeatures(user_emb_fc)(user_emb_in)

        item_emb_fc = [x for x in num_fcs if x.name == 'titleEditedVectorBert'][0]
        item_emb_in = {item_emb_fc.name: num_inputs[item_emb_fc.name]}
        item_emb = keras.layers.DenseFeatures(item_emb_fc)(item_emb_in)

        cosine = keras.layers.Dot(axes=1, normalize=True)([user_emb, item_emb])
        num_dense_feats = keras.layers.DenseFeatures(num_fcs)(num_inputs)
        num_dense_feats = keras.layers.Concatenate()([num_dense_feats, cosine])
        print("连续特征-使用=>", num_fea_use_names)
        print("连续特征-不用=>", num_fea_no_use_names)
        return num_dense_feats, num_inputs, num_cat_fcs, num_cat_inputs

    def build_cat_input_v2(self, num_cat_fcs, num_cat_inputs):
        cat_inputs = {}
        cat_fcs = []
        cat_fcs_ident = []
        cat_fea_names = []
        for use, name, size, buckets, weight_fc, default, dtype in self._cat_feat_props:
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

        print("分类特征-使用=>", cat_fea_names)
        cat_inputs = {**cat_inputs, **num_cat_inputs}
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['age', 'cityType'], hash_bucket_size=100), dimension=5))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['age', 'province', 'category'], hash_bucket_size=10000), 20))
        cat_fcs_ident.append(fc.embedding_column(fc.crossed_column(['age', 'dayOfMonth'], hash_bucket_size=70), 5))
        cat_fcs_ident.append(fc.embedding_column(fc.crossed_column(['age', 'dayOfWeek'], hash_bucket_size=70), 5))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['age', 'lastClickedArticleCategory'], hash_bucket_size=2000), 15))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['age', 'lastClickedArticleSubCategory'], hash_bucket_size=15000),
                                30))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['age', 'lastClickedHour'], hash_bucket_size=250), 5))
        cat_fcs_ident.append(fc.embedding_column(fc.crossed_column(['age', 'model'], hash_bucket_size=1500), 10))
        cat_fcs_ident.append(fc.embedding_column(fc.crossed_column(['age', 'sex'], hash_bucket_size=30), 5))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['age', 'source', 'sex'], hash_bucket_size=3000), 15))
        cat_fcs_ident.append(fc.embedding_column(fc.crossed_column(['age', 'subCategory'], hash_bucket_size=1300), 10))

        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['dayOfWeek', 'category'], hash_bucket_size=250), 5))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['dayOfWeek', 'subCategory'], hash_bucket_size=1000), 10))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['dayOfWeek', 'source'], hash_bucket_size=570), 5))

        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['weekOfMonth', 'category'], hash_bucket_size=250), 10))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['weekOfMonth', 'subCategory'], hash_bucket_size=1000), 10))
        cat_fcs_ident.append(
            fc.embedding_column(fc.crossed_column(['weekOfMonth', 'source'], hash_bucket_size=570), 5))

        cat_dense_feats_emb = keras.layers.DenseFeatures(cat_fcs_ident)(cat_inputs)
        cat_dense_feats = tf.stack(cat_fcs, axis=1)

        return cat_dense_feats, cat_dense_feats_emb, cat_inputs

    def build(self):
        hidden_units = [128, 64, 32]
        dropout = 0.2
        num_dense_feats, num_inputs, num_cat_fcs, num_cat_inputs = self.build_num_inputs()
        cat_dense_feats, cat_dense_feats_ident, cat_inputs = self.build_cat_input_v2(num_cat_fcs=num_cat_fcs,
                                                                                     num_cat_inputs=num_cat_inputs)

        lr_input_layer = tf.keras.layers.Concatenate(axis=1)([num_dense_feats, cat_dense_feats_ident])
        lr_input_layer = tf.keras.layers.Dense(units=1, name="LinearLayer")(lr_input_layer)

        fm_input_layer = tf.keras.layers.Lambda(self.fm_cross, name="FmCrossLayer")(cat_dense_feats)

        input_dnn_layer = tf.keras.layers.Flatten()(cat_dense_feats)
        for hidden_unit in hidden_units:
            input_dnn_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)(
                input_dnn_layer)
            input_dnn_layer = tf.keras.layers.Dropout(rate=dropout)(input_dnn_layer)
        dnn_logits_layer = tf.keras.layers.Dense(units=1, activation=None)(input_dnn_layer)

        predict = tf.keras.layers.Add()(inputs=[lr_input_layer, fm_input_layer, dnn_logits_layer])
        predict = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                        kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(predict)
        model = tf.keras.models.Model(inputs={**num_inputs, **cat_inputs}, outputs=[predict])
        loss = keras.losses.binary_crossentropy

        return model, loss


class FineTuneV1(ModelBuilderBase):
    def build(self):
        self.model = tf.keras.models.load_model(self._finetune_model)
        self.model.summary()


class DinBuider(ModelBuilderBase):
    def query_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)
        # query_names = ['category', 'subCategory']
        query_names = ['item_id', 'cat_id']
        for name in query_names:
            id_lookup_idx = sparse_input_dict[name]
            group_embedding_dict['default_group'].append(sparse_embedding_dict[name](id_lookup_idx))
        return list(chain.from_iterable(group_embedding_dict.values()))

    def keys_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)
        keys_names = ['category', 'subCategory']
        keys_names = ['item_id', 'cat_id']
        for name in keys_names:
            # id_lookup_idx = sparse_input_dict[name + '_his']
            id_lookup_idx = sparse_input_dict['his_'+name]
            group_embedding_dict['default_group'].append(sparse_embedding_dict[name](id_lookup_idx))
        return list(chain.from_iterable(group_embedding_dict.values()))

    def dnn_input_embedding_lookup(self, sparse_embedding_dict, sparse_input_dict):
        group_embedding_dict = defaultdict(list)
        # for use, name, size, buckets, weight_fc, default, dtype in self._cat_feat_props:
        #     if dtype != tf.string or name in ['userId', 'category_his', 'subCategory_his']:
        #         continue
        #     id_lookup_idx = sparse_input_dict[name]
        #     group_embedding_dict['default_group'].append(sparse_embedding_dict[name](id_lookup_idx))

        user_lookup_idx = sparse_input_dict["user"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["user"](user_lookup_idx))
        gender_lookup_idx = sparse_input_dict["gender"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["gender"](gender_lookup_idx))
        item_id_lookup_idx = sparse_input_dict["item_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["item_id"](item_id_lookup_idx))
        cat_id_lookup_idx = sparse_input_dict["cat_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["cat_id"](cat_id_lookup_idx))
        return list(chain.from_iterable(group_embedding_dict.values()))

    def get_dense_input(self, features):
        dense_input_list = []
        # for use, name, size, mean, stddev, normalize, default, dtype, in self._num_feat_props:
        #     if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight':
        #         continue
        #     dense_input_list.append(features[name])
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

    def buildInput(self):
        input_features = {}
        sparse_embedding = {}
        l2_reg = 1e-6

        input_features["user"] = keras.Input(
            shape=(1,), name="user", dtype=tf.int32)
        input_features["gender"] = keras.Input(
            shape=(1,), name="gender", dtype=tf.int32)
        input_features["item_id"] = keras.Input(
            shape=(1,), name="item_id", dtype=tf.int32)
        input_features["cat_id"] = keras.Input(
            shape=(1,), name="cat_id", dtype=tf.int32)
        input_features["pay_score"] = keras.Input(
            shape=(1,), name="pay_score", dtype=tf.float32)
        input_features["his_item_id"] = keras.Input(
            shape=(4,), name="his_item_id", dtype=tf.int32)
        input_features["seq_length"] = keras.Input(
            shape=(1,), name="seq_length", dtype=tf.int32)
        input_features["his_cat_id"] = keras.Input(
            shape=(4,), name="his_cat_id", dtype=tf.int32)

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
                               name='sparse_emb_cat_id')
        cat_id_emb.trainable = True
        sparse_embedding["cat_id"] = cat_id_emb

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
                                   name='sparse_seq_emb_hist_cat_id',
                                   mask_zero=True)
        his_cat_id_emb.trainable = True
        sparse_embedding["cat_id"] = his_cat_id_emb

        # for use, name, size, mean, stddev, normalize, default, dtype, in self._num_feat_props:
        #     if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight':
        #         continue
        #     shape = (size,)
        #     if name in ['category','subCategory','category_his','subCategory_his']:
        #         dtype = tf.int64
        #     input_features[name] = keras.layers.Input(name=name, shape=shape, dtype=dtype)
        #
        # for use, name, size, buckets, weight_fc, default, dtype in self._cat_feat_props:
        #     if dtype != tf.string or name == 'userId':
        #         continue
        #     shape = (size,)
        #     if name in ['category','subCategory','category_his','subCategory_his']:
        #         dtype = tf.int32
        #     input_features[name] = keras.layers.Input(name=name, shape=shape, dtype=dtype)
        #     cur_emb = Embedding(size, 10,
        #                         embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
        #                         embeddings_regularizer=l2(l2_reg),
        #                         name='sparse_emb_' + name)
        #     cur_emb.trainable = True
        #     sparse_embedding[name] = cur_emb

        return input_features, sparse_embedding

    def build(self):

        input_features, embDic = self.buildInput()
        inputs_list = list(input_features.values())
        query_emb_list = self.query_embedding_lookup(embDic, input_features)
        keys_emb_list = self.keys_embedding_lookup(embDic, input_features)
        dnn_input_emb_list = self.dnn_input_embedding_lookup(embDic, input_features)
        dense_value_list = self.get_dense_input(input_features)

        query_emb = self.concat_func(query_emb_list, mask=True)
        keys_emb = self.concat_func(keys_emb_list, mask=True)
        deep_input_emb = self.concat_func(dnn_input_emb_list)

        hist = dinUnit((80, 40), 'dice', weight_normalization=False, supports_masking=True)([
            query_emb, keys_emb])

        deep_input_emb = tf.keras.layers.Concatenate()([deep_input_emb, hist])
        dnn_input = self.combined_dnn_input([deep_input_emb], dense_value_list)
        output = DNN((256, 128, 64), 'relu', 0, 0, False, seed=1024)(dnn_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False)(output)
        output = PredictionLayer('binary')(final_logit)
        model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
        loss = keras.losses.binary_crossentropy
        return model, loss


def main():
    from lib.feat.parser import FeatParser

    feat_parser = FeatParser(rp=r'feat_br.tsv')
    num_feat_props = feat_parser.get_num_feat()
    cat_feat_props = feat_parser.get_cat_feat()

    mb = DinBuider(ckpt_dir='', log_dir='', num_feat_props=num_feat_props, cat_feat_props=cat_feat_props)
    model, loss = mb.build()
    model.summary()


if __name__ == '__main__':
    main()
