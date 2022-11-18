import tensorflow as tf
from parseFeatSchema import din_fc_parserTsv

sess = tf.compat.v1.get_default_session()
import tensorflow.keras.backend as K


def getFeatParserInfo():
    # feat_parser = v14_5_7parserTsv.FeatParser(rp=r'E:\gitWork\tfGatling\src\tfDataSet\v14_5_7_feat_br.tsv',
    #                                    mean_stddev_path=r'E:\gitWork\tfGatling\src\tfDataSet\v14_5_7_mean_stddev.tsv')
    feat_parser = din_fc_parserTsv.FeatParser(rp=r'E:\gitWork\tfGatling\src\tfDataSet\din_feat_br.tsv',
                                              mean_stddev_path=None)
    print('num feat')
    print('\n'.join(str(x) for x in feat_parser.get_num_feat()))
    print('cat feat')
    print('\n'.join(str(x) for x in feat_parser.get_cat_feat()))
    print('seq feat')
    print('\n'.join(str(x) for x in feat_parser.get_seq_feat()))

    num_feat_tsv = feat_parser.get_num_feat()
    cat_feat_tsv = feat_parser.get_cat_feat()
    seq_feat_tsv = feat_parser.get_seq_feat()

    feat_props = [(featName, featSize, featDtype, featDefaultV) for _, featName, featSize, *_, featDefaultV, featDtype
                  in
                  num_feat_tsv + cat_feat_tsv + seq_feat_tsv]
    return feat_props, num_feat_tsv, cat_feat_tsv, seq_feat_tsv


pass

feat_props, _, _, seq_props = getFeatParserInfo()


def tf_record_parser():
    # features = {
    #     "weekOfMonth": tf.io.FixedLenFeature(dtype=tf.int64,
    #                                          shape=(1,),
    #                                          default_value=0),
    #     "cityType": tf.io.FixedLenFeature(dtype=tf.int64,
    #                                       shape=(1,),
    #                                       default_value=0),
    #     "lastVersion": tf.io.FixedLenFeature(dtype=tf.int64,
    #                                          shape=(1,),
    #                                          default_value=0),
    #     "label": tf.io.FixedLenFeature(dtype=tf.float32,
    #                                    shape=(1,),
    #                                    default_value=0.0)
    # }
    # features = {name: tf.io.FixedLenFeature(dtype=dtype,
    #                                         shape=(1,) if size == 1 else (size,),
    #                                         default_value=default if size == 1 else [default] * size)
    #             for name, size, dtype, default in feat_props}
    features = {"firstVersion": tf.io.FixedLenFeature(dtype=tf.int64,
                                                      shape=(1,),
                                                      default_value=0)}
    featrueSeq = {"category_his": tf.io.FixedLenSequenceFeature(shape=(),
                                                                dtype=tf.int64,
                                                                allow_missing=True,
                                                                default_value=0)}
    pass

    # tf records parser
    def parser(rec):
        # context_parsed, sequence_parsed = tf.io.parse_sequence_example(serialized=rec,
        #                                                                       context_features=features,
        #                                                                       sequence_features=featrueSeq)
        # feats = {k: v for k, v in sequence_parsed.items()}
        # context_feats = {k: v for k, v in context_parsed.items() if k != 'label'}
        # label = context_feats['label']
        # return feats,label
        parsed = tf.io.parse_example(rec, {**features, **featrueSeq})
        # label = parsed['categorySeqs']
        # tf.print(label.shape)
        # tf.print(label)
        # feats = {k: v for k, v in parsed.items() if k != 'label'}

        # print(K.get_value(parsed)) 没用，没有输出

        return parsed

    return parser


def tf_seq_record_parser():
    features = {name: tf.io.FixedLenFeature(dtype=dtype,
                                            shape=(1,) if size == 1 else (size,),
                                            default_value=default if size == 1 else [default] * size)
                for name, size, dtype, default in feat_props}
    featrueSeq = {name: tf.io.FixedLenSequenceFeature(shape=(),
                                                      dtype=dtype,
                                                      allow_missing=True,
                                                      default_value=None)
                  for _, name, size, *_, default, dtype in seq_props}
    pass

    # tf records parser
    def parser(rec):
        sequence_parsed = tf.io.parse_sequence_example(serialized=rec,
                                                       context_features=features,
                                                       sequence_features=featrueSeq)
        return sequence_parsed
        # feats = {k: v for k, v in sequence_parsed.items()}
        # context_feats = {k: v for k, v in context_parsed.items() if k != 'label'}
        # label = context_feats['label']
        # return feats,label

        # parsed = tf.io.parse_example(rec, {**features, **featrueSeq})
        # feats = {k: v for k, v in parsed.items() if k != 'label'}
        # label = parsed['label']
        # return feats, label

    return parser


def tf_seq_record_single_parser():
    features = {name: tf.io.FixedLenFeature(dtype=dtype,
                                            shape=(1,) if size == 1 else (size,),
                                            default_value=default if size == 1 else [default] * size)
                for name, size, dtype, default in feat_props}
    featrueSeq = {name: tf.io.FixedLenSequenceFeature(shape=[1],
                                                      dtype=dtype,
                                                      allow_missing=True,
                                                      default_value=None)
                  for _, name, size, *_, default, dtype in seq_props}
    pass

    # tf records parser
    def parser(rec):
        sequence_parsed = tf.io.parse_single_sequence_example(serialized=rec,
                                                              context_features=features,
                                                              sequence_features=featrueSeq)
        return sequence_parsed
        # feats = {k: v for k, v in sequence_parsed.items()}
        # context_feats = {k: v for k, v in context_parsed.items() if k != 'label'}
        # label = context_feats['label']
        # return feats,label

        # parsed = tf.io.parse_example(rec, {**features, **featrueSeq})
        # feats = {k: v for k, v in parsed.items() if k != 'label'}
        # label = parsed['label']
        # return feats, label

    return parser


def data_gen_seq(fs, batch_size):
    ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=16)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(map_func=tf_seq_record_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=batch_size * 4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def data_gen_seq_byone(fs):
    ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=16)

    ds = ds.map(map_func=tf_seq_record_single_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def data_gen(fs, batch_size):
    ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=16)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(map_func=tf_record_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=batch_size * 4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def data_gen_byone(fs):
    ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=16)
    ds = ds.map(map_func=tf_record_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
