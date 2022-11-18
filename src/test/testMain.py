import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

ceshi = map(lambda x:x+'_his',['category', 'subcategory', 'source'])
print(list(ceshi))
def sequence_categorical_hash_column():
    # 用法同categorical_column_with_hash_bucket完全一致
    column = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        key="feature",
        hash_bucket_size=5000,
        dtype=tf.string)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [[["value1"], ["value2"]], [["value3"], ["value3"]]],
            [[["value3"], ["value5"]], [["value4"], ["value4"]]]
        ])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
columnRs = sequence_categorical_hash_column()
print(columnRs)