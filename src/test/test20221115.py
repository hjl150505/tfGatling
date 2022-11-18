import tensorflow as tf
import numpy as np

"""
测试 写入FixedLenSequenceFeature 和 parse_sequence_example
测试 读取FixedLenSequenceFeature 和 parse_sequence_example
"""

# tf.enable_eager_execution()

var_int = [1, 2, 3, 4, 5, 1, 2, 3, 4]  # 整数
var_float = np.random.randn(2, 3).astype(np.float)  # 浮点数
var_str = list(set('12'))  # 字符串
var_len_int = [[1], [2] * 2]  # 变长Feature数据
var_seq_int = [[1, 2, 3], [4, 5]]  # 不定长的序列

serialized_examples = []
serialized_sequence_examples = []
for i in range(2):
    int_data = var_int
    float_data = var_float[i]
    str_data = var_str[i]
    len_data = var_len_int[i]
    seq_data = var_seq_int[i]
    int_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=int_data))
    float_feature = tf.train.Feature(float_list=tf.train.FloatList(value=float_data))
    str_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(s, encoding='utf8') for s in str_data]))
    len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=len_data))
    seq_feature = tf.train.FeatureList(
        feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[v])) for v in seq_data])
    n_seq_features = {
        "int_feature": int_feature,
        "float_feature": float_feature,
        "str_feature": str_feature,
        "len_feature": len_feature}
    context_features = tf.train.Features(feature=n_seq_features)
    sequence_feature = {
        "seq_feature": seq_feature
    }
    sequence_features = tf.train.FeatureLists(feature_list=sequence_feature)
    example = tf.train.Example(features=context_features)
    sequence_example = tf.train.SequenceExample(context=context_features, feature_lists=sequence_features)
    # print(example)
    serialized_example = example.SerializeToString()
    serialized_sequence_example = sequence_example.SerializeToString()
    serialized_examples.append(serialized_example)
    serialized_sequence_examples.append(serialized_sequence_example)

context_features = {
    'int_feature': tf.io.FixedLenFeature([9], dtype=tf.int64),
    'float_feature': tf.io.FixedLenFeature([3], dtype=tf.float32),
    'str_feature': tf.io.FixedLenFeature([1], dtype=tf.string),
    'len_feature': tf.io.VarLenFeature(dtype=tf.int64)
}

context_parsed = tf.io.parse_example(serialized=serialized_examples, features=context_features)
tf.print('parse_example解析之后的结果')
tf.print('context_parsed', context_parsed, summarize=-1)



sequence_features = {
    'seq_feature': tf.io.FixedLenSequenceFeature([1], dtype=tf.int64,allow_missing=True,default_value=None)
}
context_parsed, sequence_parsed, _ = tf.io.parse_sequence_example(serialized=serialized_sequence_examples,
                                                                  context_features=context_features,
                                                                  sequence_features=sequence_features)
tf.print('parse_sequence_example解析之后的结果')

tf.print('context_parsed', context_parsed, summarize=-1)
tf.print('sequence_parsed', sequence_parsed, summarize=-1)


#######################################  测试 ####################################
context_parsed = tf.io.parse_example(serialized=serialized_examples, features={**context_features, **sequence_features})
tf.print('allFeat=>parse_example解析之后的结果')
tf.print('allFeat=>context_parsed', context_parsed, summarize=-1)
