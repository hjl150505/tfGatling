import tensorflow as tf
from tensorflow import keras, feature_column as fc
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.feature_column.sequence_feature_column import SequenceFeatures

"""
python：3.6
TensorFlow：2.1.0
执行成功
使用spark生成的tfrecord数据
"""


def tf_record_parser():
    features = {"category": tf.io.FixedLenFeature(dtype=tf.string,
                                                  shape=(1,),
                                                  default_value=""),
                "label": tf.io.FixedLenFeature(dtype=tf.float32,
                                               shape=(1,),
                                               default_value=0)}
    featrueSeq = {"feedClickCatList": tf.io.FixedLenSequenceFeature(shape=(),
                                                                    dtype=tf.string,
                                                                    allow_missing=True,
                                                                    default_value=None)}

    def parser_seq(rec):
        context_parsed, sequence_parsed, _ = tf.io.parse_sequence_example(serialized=rec,
                                                                          context_features=features,
                                                                          sequence_features=featrueSeq)
        feats = {k: v for k, v in sequence_parsed.items()}
        # context_feats = {k: v for k, v in context_parsed.items() if k != 'label'}
        # tf.print(feats)
        label = context_parsed['label']
        seqFea = feats['feedClickCatList']
        return seqFea, label

    def parser_context(rec):
        context_parsed = tf.io.parse_example(serialized=rec, features={**features, **featrueSeq})

        context_feats = {k: v for k, v in context_parsed.items() if k != 'label'}
        label = context_parsed['label']
        return context_feats, label

    return parser_context


def data_gen(fs, batch_size):
    ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=16)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(map_func=tf_record_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=batch_size * 4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


t = data_gen("F:\\data\\tensorflow\\v23_2_1\\date=20221113\\train\\part-r-00000", 4)

# test_op = tf.compat.v1.data.make_one_shot_iterator(t)
# one_element = test_op.get_next()
# idex = 0
# while (idex < 100):
#     test_op.next()
#     one_element = test_op.get_next()
#     print("one_element=>", one_element)

name = "feedClickCatList"
dtype=tf.string
num_inputs={}
bucket=50
input_ = keras.layers.Input(name=name, shape=(None,), dtype=dtype)
print("name=>"+name)
num_inputs[name]=input_
feat_col = fc.sequence_categorical_column_with_hash_bucket(key=name, hash_bucket_size=bucket, dtype=dtype)
seq_emb = fc.embedding_column(categorical_column=feat_col, dimension=10)
sequence_input = SequenceFeatures([seq_emb], trainable=True)
seq_feat_emb, seq_fea_len = sequence_input({name: input_})

model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[seq_feat_emb])
print("name=>" + name)
preVelue= model.predict(t)
print(preVelue)
# preValueDecode = [preVelue[i][0].decode(encoding='UTF-8', errors='strict') for i in range(len(preVelue[0][0]))]

# print(preValueDecode)
