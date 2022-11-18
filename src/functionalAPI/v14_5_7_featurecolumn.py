import tensorflow as tf
from tensorflow import keras, feature_column as fc
from tensorflow.python.keras.regularizers import l1_l2
from tfDataSet import v14_5_7
import numpy as np


class FmBuider():
    def __init__(self):
        self.feat_props, self.num_feat, self.cat_feat = v14_5_7.getFeatParserInfo()

    """
    python:3.6
    tf:2.1.0
    训练成功
    加载模型成功
    """

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
        return num_dense_feats, num_inputs

    def build_cat_input_v2(self):
        cat_inputs = {}
        cat_fcs = []
        cat_fcs_ident = []
        cat_fea_names = []
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

        print("分类特征-使用=>", cat_fea_names)
        cat_inputs = {**cat_inputs}
        cat_dense_feats_emb = keras.layers.DenseFeatures(cat_fcs_ident)(cat_inputs)
        cat_dense_feats = tf.stack(cat_fcs, axis=1)

        return cat_dense_feats, cat_dense_feats_emb, cat_inputs

    def build(self):
        hidden_units = [64, 32]
        dropout = 0.2
        num_dense_feats, num_inputs = self.build_num_inputs()
        cat_dense_feats, cat_dense_feats_ident, cat_inputs = self.build_cat_input_v2()

        lr_input_layer = tf.keras.layers.Concatenate(axis=1)([num_dense_feats, cat_dense_feats_ident])
        lr_input_layer = tf.keras.layers.Dense(units=1, name="LinearLayer")(lr_input_layer)

        fm_input_layer = tf.keras.layers.Lambda(self.fm_cross, name="FmCrossLayer")(cat_dense_feats)

        input_dnn_layer = tf.keras.layers.Flatten()(cat_dense_feats_ident)
        for hidden_unit in hidden_units:
            input_dnn_layer = tf.keras.layers.Dense(units=hidden_unit, activation=tf.keras.activations.relu)(
                input_dnn_layer)
            input_dnn_layer = tf.keras.layers.Dropout(rate=dropout)(input_dnn_layer)
        dnn_logits_layer = tf.keras.layers.Dense(units=1, activation=None)(input_dnn_layer)

        predict = tf.keras.layers.Add()(inputs=[lr_input_layer, fm_input_layer, dnn_logits_layer])

        predict = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                        kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(predict)
        model = tf.keras.models.Model(inputs={**num_inputs, **cat_inputs}, outputs=[predict])
        # model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[predict])
        loss = keras.losses.binary_crossentropy

        return model, loss


def trainMain():
    x = v14_5_7.data_gen(["F:\\data\\tensorflow\\v14_5_7\\date=20221111\\train\\part-r-00098",
                          "F:\\data\\tensorflow\\v14_5_7\\date=20221111\\train\\part-r-00099"], 4)
    v = v14_5_7.data_gen(["F:\\data\\tensorflow\\v14_5_7\\date=20221111\\val\\part-r-00019","F:\\data\\tensorflow\\v14_5_7\\date=20221111\\val\\part-r-00018"], 4)
    t = v14_5_7.data_gen("F:\\data\\tensorflow\\v14_5_7\\date=20221111\\test\\part-r-00016", 4)
    test_op = tf.compat.v1.data.make_one_shot_iterator(x)
    one_element = test_op.get_next()
    print(one_element)
    print("one_element=>", one_element)
    embModel = FmBuider()
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
    model.save("v14_5_7_model")

    loadModel = keras.models.load_model("v14_5_7_model")
    preRs = loadModel.predict(t)
    print(preRs)


def checkNumFcData():
    num_inputs = {}

    def norm(mean_, stddev_):
        def scale(x):
            return (x - mean_) / stddev_

        return scale

    t = v14_5_7.data_gen("F:\\data\\tensorflow\\v14_5_7\\date=20221111\\train\\part-r-00099", 4)
    _, num_feat, _ = v14_5_7.getFeatParserInfo()
    for use, name, size, mean, stddev, normalize, default, dtype, in num_feat:
        if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight' or name!="pushClickedNumByDayOfWeekLast28d":
            continue
        if name =="pushClickedNumByDayOfWeekLast28d":
            print("name=>"+name)
        shape = (1,) if size == 1 else (size,)
        input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
        num_inputs[name] = input_
        normalizer_fn = norm(mean, stddev) if normalize else None
        feat_col = fc.numeric_column(key=name, shape=shape, normalizer_fn=normalizer_fn)
        num_dense_feats = keras.layers.DenseFeatures(feat_col)(num_inputs)
        model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[num_dense_feats])
        print("name=>" + name)
        preValue = model.predict(t)
        print('特征{} 是否有nan值 {}'.format(name, np.isnan(preValue).any()))
        print(preValue.shape)
        print(preValue)
        pass


def checkCatFcData():
    num_inputs = {}

    def norm(mean_, stddev_):
        def scale(x):
            return (x - mean_) / stddev_

        return scale

    t = v14_5_7.data_gen("F:\\data\\tensorflow\\v14_5_7\\date=20221111\\train\\part-r-00099", 4)
    *_, cat_feat = v14_5_7.getFeatParserInfo()
    for use, name, size, buckets, weight_fc, default, dtype in cat_feat:
        if dtype != tf.string or name == 'userId':
            continue
        shape = (1,) if size == 1 else (size,)
        input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
        num_inputs[name] = input_
        feat_col = fc.categorical_column_with_hash_bucket(key=name, hash_bucket_size=buckets)
        if size > 1:
            feat_col = fc.weighted_categorical_column(categorical_column=feat_col, weight_feature_key=weight_fc)
        dimension = 10
        cat_fc = fc.embedding_column(categorical_column=feat_col, dimension=dimension, combiner='sum')
        num_dense_feats = keras.layers.DenseFeatures(cat_fc)(num_inputs)
        model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[num_dense_feats])
        print("name=>" + name)
        preValue = model.predict(t)
        print('特征{} 是否有nan值 {}', name, np.isnan(preValue).any())
        print(preValue.shape)
        print(preValue)
        pass


if __name__ == "__main__":
    # trainMain()
    checkNumFcData()
    pass
