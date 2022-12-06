import tensorflow as tf
from tensorflow.python.estimator.canned import metric_keys


# 模型训练结束导出最后一份模型
def model_final_exporter(model_name, feature_schema):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_schema)
    return tf.estimator.FinalExporter(name=model_name,
                                      serving_input_receiver_fn=serving_input_receiver_fn)


# 评估效果超过之前所有已存在的模型效果，就导出模型
def model_best_exporter(model_name, feature_schema, assets_extra=None, exports_to_keep=1, metric_key=metric_keys.MetricKeys.LOSS, big_better=False):
    # serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    #     feature_schema)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_schema)

    def compare(best_eval_result, current_eval_result):
        if not best_eval_result or metric_key not in best_eval_result:
            raise ValueError(
                'best_eval_result cannot be empty or no loss is found in it.')

        if not current_eval_result or metric_key not in current_eval_result:
            raise ValueError(
                'current_eval_result cannot be empty or no loss is found in it.')

        if big_better:
            return best_eval_result[metric_key] > current_eval_result[metric_key]
        else:
            return best_eval_result[metric_key] < current_eval_result[metric_key]

    return tf.estimator.BestExporter(name=model_name,
                                     serving_input_receiver_fn=serving_input_receiver_fn,
                                     compare_fn=compare,
                                     assets_extra=assets_extra,# 用于保存所需要的的额外信息，例如在nlp中的词汇表，格式：{"vocab.txt": FLAGS.vocab_file}
                                     exports_to_keep=exports_to_keep)


# 每次评估都导出模型，默认最多保存3份
def model_latest_exporter(model_name, feature_schema, exports_to_keep=3):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_schema)
    return tf.estimator.LatestExporter(name=model_name,
                                       exports_to_keep=exports_to_keep,
                                       serving_input_receiver_fn=serving_input_receiver_fn)
