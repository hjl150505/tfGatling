import tensorflow as tf
from random import randint
from random import random
import numpy as np


def genTfV1():
    # TensorFlow2.x
    writer = tf.io.TFRecordWriter("./tfrecord")

    sampleNum = 5000
    fixLen = 4
    for num in range(0, sampleNum):
        hisLen = randint(1, fixLen)
        labels = np.random.randint(0, 2, 1)
        his_item_id = set()
        his_cat_id = set()
        while len(his_item_id) < hisLen:
            his_item_id.add(randint(0, 100))
        his_item_id = list(his_item_id)

        while len(his_cat_id) < hisLen:
            his_cat_id.add(randint(0, 10))
        his_cat_id = list(his_cat_id)

        his_item_pend = [0] * (fixLen - len(his_item_id))
        his_item_id.extend(his_item_pend)
        his_cat_id.extend(his_item_pend)

        example = tf.train.Example(
            features=tf.train.Features(feature={
                "user": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(0, 100)])),
                "item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(0, 100)])),
                "gender": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(1, 2)])),
                "cat_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(0, 10)])),
                "pay_score": tf.train.Feature(float_list=tf.train.FloatList(value=[random()])),
                "his_item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=his_item_id)),
                "his_cat_id": tf.train.Feature(int64_list=tf.train.Int64List(value=his_cat_id)),
                "seq_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[hisLen])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            })
        )
        writer.write(example.SerializeToString())

    writer.close()


def genTfSeq():
    # TensorFlow2.x
    writer = tf.io.TFRecordWriter("./SeqTfrecord")
    serialized_sequence_examples = []
    sampleNum = 5000
    fixLen = 4
    seqMaxLen = 8
    var_array_int = 9
    var_float = np.random.randn(sampleNum, 6).astype(np.float)  # 浮点数
    var_str = list(set('abcdefghigklmnopqrstuvwxyz'))  # 字符串
    var_str_len = len(var_str)-1
    var_str_feat_len = 5
    for num in range(0, sampleNum):
        array_int_feat = set()
        array_str_feat = []
        hisLen = randint(1, fixLen)
        seqLen = randint(1, seqMaxLen)
        labels = np.random.randint(0, 2, 1)
        his_item_id = set()
        his_cat_id = set()
        seq_feat_cur = set()
        while len(array_int_feat) < var_array_int:
            array_int_feat.add((randint(0, 100)))
        while len(array_str_feat) < var_str_feat_len:
            array_str_feat.append(var_str[randint(0, var_str_len)])

        while len(seq_feat_cur) < seqLen:
            seq_feat_cur.add(var_str[randint(0, var_str_len)])
        seq_feat_cur = list(seq_feat_cur)
        while len(his_item_id) < hisLen:
            his_item_id.add(randint(0, 100))
        his_item_id = list(his_item_id)

        while len(his_cat_id) < hisLen:
            his_cat_id.add(randint(0, 10))
        his_cat_id = list(his_cat_id)

        his_item_pend = [0] * (fixLen - len(his_item_id))
        his_item_id.extend(his_item_pend)
        his_cat_id.extend(his_item_pend)
        seq_feat_cur_pend = [""] * (seqMaxLen - len(seq_feat_cur))
        seq_feat_cur.extend(seq_feat_cur_pend)

        context_features = tf.train.Features(feature={
            "user": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(0, 100)])),
            "item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(0, 100)])),
            "gender": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(1, 2)])),
            "cat_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[randint(0, 10)])),
            "pay_score": tf.train.Feature(float_list=tf.train.FloatList(value=[random()])),
            "his_item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=his_item_id)),
            "his_cat_id": tf.train.Feature(int64_list=tf.train.Int64List(value=his_cat_id)),
            "seq_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[hisLen])),
            "str_array_fea": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[bytes(s, encoding='utf8') for s in array_str_feat])),
            "int_array_fea": tf.train.Feature(int64_list=tf.train.Int64List(value=array_int_feat)),
            "float_array_fea": tf.train.Feature(float_list=tf.train.FloatList(value=var_float[num])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        })

        seq_feature = tf.train.FeatureList(
            feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(v, encoding='utf8')])) for v in
                     seq_feat_cur]
        )
        sequence_feature = {
            "seq_feature": seq_feature
        }
        sequence_features = tf.train.FeatureLists(feature_list=sequence_feature)
        sequence_example = tf.train.SequenceExample(context=context_features, feature_lists=sequence_features)
        serialized_sequence_example = sequence_example.SerializeToString()
        serialized_sequence_examples.append(serialized_sequence_example)
        writer.write(serialized_sequence_example)

    writer.close()
    if 0:
        context_feat_schema = {
            "user" : tf.io.FixedLenFeature([1],dtype=tf.int64),
            "item_id" :tf.io.FixedLenFeature([1],dtype=tf.int64),
            "gender":tf.io.FixedLenFeature([1],dtype=tf.int64),
            "cat_id":tf.io.FixedLenFeature([1],dtype=tf.int64),
            "pay_score":tf.io.FixedLenFeature([1],dtype=tf.float32),
            "his_item_id":tf.io.FixedLenFeature([4],dtype=tf.int64),
            "his_cat_id":tf.io.FixedLenFeature([4],dtype=tf.int64),
            "seq_length":tf.io.FixedLenFeature([1],dtype=tf.int64),
            "str_array_fea":tf.io.FixedLenFeature([5],dtype=tf.string),
            'int_array_fea': tf.io.FixedLenFeature([9], dtype=tf.int64),
            'float_array_fea': tf.io.FixedLenFeature([6], dtype=tf.float32),
            'label': tf.io.FixedLenFeature([1], dtype=tf.int64),
        }
        sequence_feat_schema = {
            'seq_feature': tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.string, allow_missing=True, default_value=None)
        }
        context_parsed, sequence_parsed, _ = tf.io.parse_sequence_example(serialized=serialized_sequence_examples,
                                                                          context_features=context_feat_schema,
                                                                          sequence_features=sequence_feat_schema)
        tf.print('context_parsed', context_parsed, summarize=-1)
        tf.print('sequence_parsed', sequence_parsed, summarize=-1)
        pass

if __name__ == "__main__":
    genTfSeq()
