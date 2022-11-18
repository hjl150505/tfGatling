import tensorflow as tf

def tf_record_parser():

    features = {
        "user":tf.io.FixedLenFeature(dtype=tf.int64,
                                            shape=(1,),
                                            default_value=0),
        "item_id": tf.io.FixedLenFeature(dtype=tf.int64,
                                        shape=(1,),
                                        default_value=0),
        "gender": tf.io.FixedLenFeature(dtype=tf.int64,
                                         shape=(1,),
                                         default_value=0),
        "cat_id": tf.io.FixedLenFeature(dtype=tf.int64,
                                        shape=(1,),
                                        default_value=0),
        "pay_score": tf.io.FixedLenFeature(dtype=tf.float32,
                                         shape=(1,),
                                         default_value=0.0),
        "his_item_id": tf.io.FixedLenFeature(dtype=tf.int64,
                                        shape=(4,),
                                        default_value=[0]*4),
        "his_cat_id": tf.io.FixedLenFeature(dtype=tf.int64,
                                             shape=(4,),
                                             default_value=[0] * 4),
        "seq_length": tf.io.FixedLenFeature(dtype=tf.int64,
                                        shape=(1,),
                                        default_value=0),
        "label": tf.io.FixedLenFeature(dtype=tf.int64,
                                           shape=(1,),
                                           default_value=0)
    }

    # tf records parser
    def parser(rec):
        parsed = tf.io.parse_example(rec, features)
        feats = {k: v for k, v in parsed.items() if k != 'label'}
        label = parsed['label']
        return feats, label

    return parser

def data_gen(fs,batch_size):
    ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=16)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(map_func=tf_record_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=batch_size * 4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds