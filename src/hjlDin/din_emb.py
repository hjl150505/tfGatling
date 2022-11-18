import tensorflow as tf
from tensorflow import keras, feature_column as fc
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.feature_column.sequence_feature_column import SequenceFeatures
from tfDataSet import din_emb_parser
tf.compat.v1.enable_eager_execution()

def checkFcData():
    num_inputs = {}

    def norm(mean_, stddev_):
        def scale(x):
            return (x - mean_) / stddev_

        return scale

    t = din_emb_parser.data_gen("F:\\data\\tensorflow\\din_emb\\date=20221025\\train\\part-r-00003",4)
    # test_op = tf.compat.v1.data.make_one_shot_iterator(t)
    # one_element = test_op.get_next()
    # idex=0
    # while(idex<100):
    #     test_op.next()
    #     one_element = test_op.get_next()
    #     print("one_element=>", one_element)
    name = "category_his"
    dtype=tf.int64
    bucket=50
    input_ = keras.layers.Input(name=name, shape=(None,), dtype=dtype)
    print("name=>"+name)
    num_inputs[name]=input_
    feat_col = fc.sequence_categorical_column_with_hash_bucket(key=name, hash_bucket_size=bucket, dtype=dtype)
    seq_emb = fc.embedding_column(categorical_column=feat_col, dimension=10)
    sequence_input = SequenceFeatures([seq_emb], trainable=True)
    seq_feat_emb, seq_fea_len = sequence_input({name: input_})

    # num_dense_feats = keras.layers.DenseFeatures(feat_col)(num_inputs)
    model = tf.keras.models.Model(inputs={**num_inputs}, outputs=[seq_feat_emb])
    print("name=>" + name)
    preVelue= model.predict(t)
    preValueDecode = [preVelue[i][0].decode(encoding='UTF-8', errors='strict') for i in range(len(preVelue[0][0]))]
    # preValueDecode = preVelue.decode(encoding='UTF-8', errors='strict')
    print(preValueDecode)



if __name__ == "__main__":
    checkFcData()
    # trainMain()
    pass
