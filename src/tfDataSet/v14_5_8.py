import tensorflow as tf
from parseFeatSchema import v14_5_8parserTsv

def getFeatParserInfo():

    feat_parser = v14_5_8parserTsv.FeatParser(rp=r'E:\gitWork\tfGatling\src\tfDataSet\v14_5_8_feat_br.tsv',
                                       mean_stddev_path=r'E:\gitWork\tfGatling\src\tfDataSet\v14_5_8_mean_stddev.tsv')
    print('num feat')
    print('\n'.join(str(x) for x in feat_parser.get_num_feat()))
    print('cat feat')
    print('\n'.join(str(x) for x in feat_parser.get_cat_feat()))

    num_feat_tsv = feat_parser.get_num_feat()
    cat_feat_tsv = feat_parser.get_cat_feat()

    feat_props = [(featName, featSize, featDtype, featDefaultV) for _, featName, featSize, *_, featDefaultV, featDtype in
                  num_feat_tsv + cat_feat_tsv]
    return feat_props,num_feat_tsv,cat_feat_tsv
pass

feat_props,_,_ = getFeatParserInfo()
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
    features = {name: tf.io.FixedLenFeature(dtype=dtype,
                                            shape=(1,) if size == 1 else (size,),
                                            default_value=default if size == 1 else [default] * size)
                for name, size, dtype, default in feat_props}
    pass

    # tf records parser
    def parser(rec):
        parsed = tf.io.parse_example(rec, features)
        feats = {k: v for k, v in parsed.items() if k != 'label'}
        label = parsed['label']
        return feats, label

    return parser


def data_gen(fs, batch_size):
    ds = tf.data.TFRecordDataset(filenames=fs, num_parallel_reads=16)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(map_func=tf_record_parser(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=batch_size * 4, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat(2)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

# def data_gen(self, batch_size=512, use_hvd=False):
#     ds_info = [('train', False, batch_size, True, True, True, "train_data"),
#                ('val', False, batch_size * 4, True, False, True, "val_data"),
#                ('test', False, batch_size * 4, False, False, False, "test_data")]
#     ds_info = [(os.path.join(os.path.join(self._rp, p), 'part-r-*'), *others) for p, *others in ds_info]
#     ds_info = [(sys_util.list_files(p), *others) for p, *others in ds_info]
#     train, val, test = [self.get_ds(fs=f, shuffle=s, batch_size=b, distribute=d, repeat=r, cache=c, use_hvd=use_hvd, sample_type=t)
#                         for f, s, b, d, r, c, t in ds_info]
#
#     return train, val, test