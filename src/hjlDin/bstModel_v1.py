import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from collections import defaultdict
from itertools import chain


class BST:
    def __init__(self,dnn_feature_columns, history_feature_list, transformer_num=1, att_head_num=8,
        use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='relu', l2_reg_dnn=0,
        l2_reg_embedding=1e-6, dnn_dropout=0.0, seed=1024, task='binary'):
        pass
    def buildInput(self):
        input_features = {}
        input_features["user"] = keras.Input(
                        shape=(1,), name="user", dtype=tf.int32)
        input_features["gender"] = keras.Input(
            shape=(1,), name="gender", dtype=tf.int32)
        input_features["item_id"] = keras.Input(
            shape=(1,), name="item_id", dtype=tf.int32)
        input_features["cat_id"] = keras.Input(
            shape=(1,), name="cat_id", dtype=tf.int32)
        input_features["pay_score"] = keras.Input(
            shape=(1,), name="pay_score", dtype=tf.float32)
        input_features["his_item_id"] = keras.Input(
            shape=(4,), name="his_item_id", dtype=tf.int32)
        input_features["seq_length"] = keras.Input(
            shape=(1,), name="seq_length", dtype=tf.int32)
        input_features["his_cat_id"] = keras.Input(
            shape=(4,), name="his_cat_id", dtype=tf.int32)
        return input_features

    def create_embedding_dict(self):
        sparse_embedding = {}
        l2_reg = 1e-6
        user_emb = Embedding(3, 10,
                        embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                        embeddings_regularizer=l2(l2_reg),
                        name='sparse_emb_user')
        user_emb.trainable = True
        sparse_embedding["user"] = user_emb

        gender_emb = Embedding(2, 4,
                             embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                             embeddings_regularizer=l2(l2_reg),
                             name='sparse_emb_gender')
        gender_emb.trainable = True
        sparse_embedding["gender"] = gender_emb

        item_id_emb = Embedding(4, 8,
                               embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                               embeddings_regularizer=l2(l2_reg),
                               name='sparse_emb_item_id')
        item_id_emb.trainable = True
        sparse_embedding["item_id"] = item_id_emb

        cat_id_emb = Embedding(3, 4,
                                embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                embeddings_regularizer=l2(l2_reg),
                                name='sparse_emb_cat_id')
        cat_id_emb.trainable = True
        sparse_embedding["cat_id"] = cat_id_emb

        his_item_id_emb = Embedding(4, 8,
                        embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                        embeddings_regularizer=l2(
                            l2_reg),
                        name='sparse_seq_emb_hist_item_id',
                        mask_zero=True)
        his_item_id_emb.trainable = True
        sparse_embedding["item_id"] = his_item_id_emb

        his_cat_id_emb = Embedding(3, 4,
                                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                    embeddings_regularizer=l2(
                                        l2_reg),
                                    name='sparse_seq_emb_hist_cat_id',
                                    mask_zero=True)
        his_cat_id_emb.trainable = True
        sparse_embedding["cat_id"] = his_cat_id_emb

        return sparse_embedding

    def query_embedding_lookup(self,sparse_embedding_dict,sparse_input_dict):
        group_embedding_dict = defaultdict(list)

        item_id_lookup_idx = sparse_input_dict["item_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["item_id"](item_id_lookup_idx))
        cat_id_lookup_idx = sparse_input_dict["cat_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["cat_id"](cat_id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def keys_embedding_lookup(self,sparse_embedding_dict,sparse_input_dict):
        group_embedding_dict = defaultdict(list)

        his_item_id_lookup_idx = sparse_input_dict["his_item_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["item_id"](his_item_id_lookup_idx))
        his_cat_id_lookup_idx = sparse_input_dict["his_cat_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["cat_id"](his_cat_id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))

    def dnn_input_embedding_lookup(self,sparse_embedding_dict,sparse_input_dict):
        group_embedding_dict = defaultdict(list)

        user_lookup_idx = sparse_input_dict["user"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["user"](user_lookup_idx))
        gender_lookup_idx = sparse_input_dict["gender"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["gender"](gender_lookup_idx))
        item_id_lookup_idx = sparse_input_dict["item_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["item_id"](item_id_lookup_idx))
        cat_id_lookup_idx = sparse_input_dict["cat_id"]
        group_embedding_dict['default_group'].append(sparse_embedding_dict["cat_id"](cat_id_lookup_idx))

        return list(chain.from_iterable(group_embedding_dict.values()))
    def get_dense_input(self,features):
        dense_input_list = []
        dense_input_list.append(features["pay_score"])
        return dense_input_list


    def build(self):
        features = self.buildInput()
        inputs_list = list(features.values())
        user_behavior_length = features["seq_length"]
        embDic = self.create_embedding_dict()
