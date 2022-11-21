import tensorflow as tf
from tensorflow import keras, feature_column as fc
from tensorflow.python.keras.regularizers import l1_l2
from tfDataSet import v14_5_7
import numpy as np
from model_exporter import model_best_exporter
from tensorflow.python.estimator.canned import metric_keys
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function
import glob

# from tensorflow_estimator.python.estimator import estimator_lib
import random
import tensorflow as tf
from tensorflow.python.framework.ops import get_name_scope
from tensorflow.python.estimator.canned import head as head_utils
import tensorflow.keras.backend as k
import sys

sys.path.append(".")
sys.path.append("..")


tf.logging.set_verbosity(tf.logging.INFO)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


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


class DeepFmEstimator(tf.estimator.Estimator):
    def __init__(self, model_dir,
                          hidden_units,
                          optimizer,
                          activation_fn,
                          dropout=None,
                          batch_norm=False,
                          weight_column=None,
                          label_vocabulary=None,
                          loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
                          params=None,
                          config=None,
                          warm_start_from=None):

        # head = head_utils.head._binary_logistic_or_multi_class_head(
        #     n_classes=2,
        #     weight_column=weight_column,
        #     label_vocabulary=label_vocabulary,
        #     loss_reduction=loss_reduction
        # )
        self.logging_hook = None

        def model_fn(features, labels, mode):
            hidden_units = [64, 32]
            dropout = 0.2
            num_dense_feats = self.build_num_inputs()
            parser_num_feat = fc.input_layer(features,num_dense_feats)
            cat_feats = self.build_cat_input()
            parser_cat_feat = fc.input_layer(features,cat_feats)
            # cat_dense_feats, cat_dense_feats_ident, cat_inputs = self.build_cat_input_v2()
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

            y_click_prediction = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid,
                                                       kernel_regularizer=l1_l2(l1=0.1, l2=0.01))(predict)

            if mode != tf.estimator.ModeKeys.PREDICT:
                # ------拆分标签------
                labels_click = tf.reshape(labels, shape=[-1, ])
                labels_click = tf.cast(labels_click, dtype=tf.float32)

            # 预测结果导出格式设置
            predictions = {
                "deep_output": tf.reshape(y_click_prediction, shape=[-1, 1], name='deep_output'),
                "click": y_click_prediction
            }
            export_outputs = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                    predictions)}
            # Estimator预测模式
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs=export_outputs)

            if mode != tf.estimator.ModeKeys.PREDICT:
                # ------拆分标签，构建损失------
                # loss = tf.reduce_mean(
                #     tf.nn.weighted_cross_entropy_with_logits(logits=y_click, targets=labels_click,
                #                                              pos_weight=FLAGS.pos_weight))
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_click_prediction)
                )
                # for wgt in wgts:
                #     loss += l2_reg * tf.nn.l2_loss(wgt)

            if mode == tf.estimator.ModeKeys.EVAL:
                # Provide an estimator spec for `ModeKeys.EVAL`
                eval_metric_ops = {
                    "auc": tf.metrics.auc(labels_click, y_click_prediction),
                    "auc_click": tf.metrics.auc(labels_click, y_click_prediction)
                }
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=loss,
                    eval_metric_ops=eval_metric_ops)

            # ------bulid optimizer------

            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6)

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            # Provide an estimator spec for `ModeKeys.TRAIN` modes
            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op)

        super().__init__(model_fn, model_dir, config, params, warm_start_from)

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
            # input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            normalizer_fn = norm(mean, stddev) if normalize else None
            feat_col = fc.numeric_column(key=name, shape=shape, normalizer_fn=normalizer_fn)

            if use:
                num_fcs.append(feat_col)
                # num_inputs[name] = input_
                num_fea_use_names.append(name)
            else:
                num_cat_fcs.append(feat_col)
                # num_cat_inputs[name] = input_
                num_fea_no_use_names.append(name)

        # user_emb_fc = [x for x in num_fcs if x.name == 'userEmbedding'][0]
        # user_emb_in = {user_emb_fc.name: num_inputs[user_emb_fc.name]}
        # user_emb = keras.layers.DenseFeatures(user_emb_fc)(user_emb_in)
        #
        # item_emb_fc = [x for x in num_fcs if x.name == 'titleEditedVectorBert'][0]
        # item_emb_in = {item_emb_fc.name: num_inputs[item_emb_fc.name]}
        # item_emb = keras.layers.DenseFeatures(item_emb_fc)(item_emb_in)
        #
        # cosine = keras.layers.Dot(axes=1, normalize=True)([user_emb, item_emb])
        # num_dense_feats = keras.layers.DenseFeatures(num_fcs)(num_inputs)
        # num_dense_feats = keras.layers.Concatenate()([num_dense_feats, cosine])
        # print("连续特征-使用=>", num_fea_use_names)
        # print("连续特征-不用=>", num_fea_no_use_names)
        return num_fcs

    def build_cat_input(self):
        cat_inputs = {}
        cat_fcs = []
        cat_fcs_ident = []
        cat_fea_names = []
        for use, name, size, buckets, weight_fc, default, dtype in self.cat_feat:
            if dtype != tf.string or name == 'userId':
                continue
            cat_fea_names.append(name)
            shape = (1,) if size == 1 else (size,)
            # input_ = keras.layers.Input(name=name, shape=shape, dtype=dtype)
            feat_col = fc.categorical_column_with_hash_bucket(key=name, hash_bucket_size=buckets)
            if size > 1:
                feat_col = fc.weighted_categorical_column(categorical_column=feat_col, weight_feature_key=weight_fc)
            dimension = 10
            cat_fc = fc.embedding_column(categorical_column=feat_col, dimension=dimension, combiner='sum')
            # cat_fcs.append(keras.layers.DenseFeatures(cat_fc)({name: input_}))
            # cat_inputs[name] = input_
            cat_fcs_ident.append(cat_fc)

        print("分类特征-使用=>", cat_fea_names)
        # cat_inputs = {**cat_inputs}
        # cat_dense_feats_emb = keras.layers.DenseFeatures(cat_fcs_ident)(cat_inputs)
        cat_dense_feats = tf.stack(cat_fcs, axis=1)

        return cat_dense_feats

    def fm_cross(self, embeddings):
        square_sum_tensor = tf.math.square(tf.math.reduce_sum(embeddings, axis=1))
        sum_square_tensor = tf.math.reduce_sum(tf.math.square(embeddings), axis=1)
        return 0.5 * tf.math.reduce_sum(square_sum_tensor - sum_square_tensor, axis=1, keepdims=True)


def trainMain():
    # x = v14_5_7.data_gen(["F:\\data\\tensorflow\\v14_5_7\\date=20221111\\train\\part-r-00098",
    #                       "F:\\data\\tensorflow\\v14_5_7\\date=20221111\\train\\part-r-00099"], 4)
    # v = v14_5_7.data_gen(["F:\\data\\tensorflow\\v14_5_7\\date=20221111\\val\\part-r-00019",
    #                       "F:\\data\\tensorflow\\v14_5_7\\date=20221111\\val\\part-r-00018"], 4)
    # t = v14_5_7.data_gen("F:\\data\\tensorflow\\v14_5_7\\date=20221111\\test\\part-r-00016", 4)
    # test_op = tf.compat.v1.data.make_one_shot_iterator(x)
    # one_element = test_op.get_next()
    # print("one_element=>", one_element)


    data_dir ="F:/data/tensorflow/v14_5_7"
    print(data_dir)
    tr_files = glob.glob("%s/train/date=20221111/*" % data_dir)
    # tr_files = get_file_list(base_path=FLAGS.data_dir, start_date_str=FLAGS.train_start_date,
    #                          end_date_str=FLAGS.train_end_date, shuffle=True)
    print("train file list:")
    print(','.join(tr_files))
    va_files = glob.glob("%s/validation_data/date=20221111/*" % data_dir)
    # va_files = get_file_list(base_path=FLAGS.data_dir, start_date_str=FLAGS.test_start_date,
    #                          end_date_str=FLAGS.test_end_date, shuffle=False)
    print("test file list:")
    print(','.join(va_files))
    # te_files = get_file_list(base_path=FLAGS.data_dir, start_date_str=FLAGS.test_start_date,
    #                         end_date_str=FLAGS.test_end_date, shuffle=False)
    # print("te_files:", te_files)

    # ------bulid Tasks------
    model_params = {
        "learning_rate": 0.001,
        "l2_reg": 0.01,
    }

    # strategy = tf.contrib.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:2","/gpu:3"])  # train_distribute=strategy, eval_distribute=strategy
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())  # train_distribute=strategy, eval_distribute=strategy
    # strategy = tf.distribute.experimental.CentralStorageStrategy(
    # )  # train_distribute=strategy, eval_distribute=strategy
    config_proto = tf.ConfigProto(allow_soft_placement=True,
                                  device_count={'GPU': 4},
                                  intra_op_parallelism_threads=0,
                                  # 线程池中线程的数量，一些独立的操作可以在这指定的数量的线程中进行并行，如果设置为0代表让系统设置合适的数值
                                  inter_op_parallelism_threads=0,
                                  # 每个进程可用的为进行阻塞操作节点准备的线程池中线程的数量，设置为0代表让系统选择合适的数值，负数表示所有的操作在调用者的线程中进行。注意：如果在创建第一个Session的适合制定了该选项，那么之后创建的所有Session都会保持一样的设置，除非use_per_session_threads为true或配置了session_inter_op_thread_pool。
                                  log_device_placement=False,
                                  # gpu_options=gpu_options
                                  )
    config = tf.estimator.RunConfig(train_distribute=strategy, eval_distribute=strategy, session_config=config_proto,
                                    log_step_count_steps=600, save_checkpoints_steps=120 * 50,
                                    save_summary_steps=600 * 50, tf_random_seed=2021)

    # Model = tf.estimator.Estimator(model_fn=model_fn, model_dir="F:\\modelRs\\v14_5_7_estimator", params=model_params,
    #                                config=config)
    Model = DeepFmEstimator(model_dir="F:\\modelRs\\v14_5_7_estimator", hidden_units=[32, 16],
                               config=config)

    batch_size = 32
    task_type = 'train'
    job_name = "estimatorModelTrain"
    servable_model_dir = ""
    if task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: train_input_fn(tr_files,
                                      batch_size=batch_size))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: train_input_fn(va_files, batch_size=batch_size),
            steps=None,
            exporters=[model_best_exporter(job_name, feature_schema=feature_schema,
                                           exports_to_keep=1,
                                           metric_key=metric_keys.MetricKeys.AUC, big_better=False)],
            start_delay_secs=100, throttle_secs=100
        )
        tf.estimator.train_and_evaluate(Model, train_spec, eval_spec)
    elif task_type == 'eval':
        Model.evaluate(input_fn=lambda: train_input_fn(va_files, batch_size=batch_size))
    elif task_type == 'infer':
        preds = Model.predict(
            input_fn=lambda: train_input_fn(va_files, batch_size=batch_size),
            predict_keys=["deep_out", "click", "play", "valid_play"])
        with open(data_dir + "/new_idea52_pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\t%f\t%f\t%f\n" % (prob['prob'], prob['click'], prob['play'], prob['valid_play']))
    elif task_type == 'export':
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_schema)
        Model.export_savedmodel(servable_model_dir, serving_input_receiver_fn)
    # loadModel = keras.models.load_model("v14_5_7_model")
    # preRs = loadModel.predict(t)
    # print(preRs)

feat_props, num_feat, cat_feat = v14_5_7.getFeatParserInfo()
def tf_record_parser():

    features = {name: tf.io.FixedLenFeature(dtype=dtype,
                                            shape=(1,) if size == 1 else (size,),
                                            default_value=default if size == 1 else [default] * size)
                for name, size, dtype, default in feat_props}
    pass

    # tf records parser
    def parser(rec):

        parsed = tf.io.parse_example(rec, features)
        sampleW = parsed['dayDecayWeight']
        feats = {k: v for k, v in parsed.items() if k != 'label'}
        label = parsed['label']

        return feats, label, sampleW

    return parser

def train_input_fn(filenames, batch_size=32):

    files = tf.data.Dataset.list_files(filenames)

    ds = tf.data.TFRecordDataset(filenames=files, num_parallel_reads=16)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(map_func=tf_record_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=batch_size * 4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

    # return dataset

def checkNumFcData():
    num_inputs = {}

    def norm(mean_, stddev_):
        def scale(x):
            return (x - mean_) / stddev_

        return scale

    t = v14_5_7.data_gen("F:\\data\\tensorflow\\v14_5_7\\date=20221111\\train\\part-r-00099", 4)
    _, num_feat, _ = v14_5_7.getFeatParserInfo()
    for use, name, size, mean, stddev, normalize, default, dtype, in num_feat:
        if dtype != tf.float32 or name == 'label' or name == 'dayDecayWeight' or name != "pushClickedNumByDayOfWeekLast28d":
            continue
        if name == "pushClickedNumByDayOfWeekLast28d":
            print("name=>" + name)
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
