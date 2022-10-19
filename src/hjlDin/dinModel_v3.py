from modelBase import ModelBuilderBase
"""
可用版本
环境：
windows10
tf2.2.3
python3.6.2
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from collections import defaultdict
from itertools import chain
from tensorflow.python.keras.layers import LSTM, Lambda, Layer
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras import backend as K
import numpy as np
from dinUnit import dinUnit

try:
    unicode
except NameError:
    unicode = str
class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Dice(Layer):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

      Input shape
        - Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.

      Output shape
        - Same shape as the input.

      Arguments
        - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).

        - **epsilon** : Small float added to variance to avoid dividing by zero.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        # tf.layers.batch_normalization(
        # inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, unicode)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer
def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)
class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

      Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape[0], input_shape[1]))
        size = 4 * \
               int(input_shape[0][-1]
                   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")
        self.dnn = DNN(self.hidden_units, self.activation, self.l2_reg, self.dropout_rate, self.use_bn, seed=self.seed)

        self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(
            x[0], x[1], axes=(-1, 0)), x[2]))

        super(LocalActivationUnit, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = self.dnn(att_input, training=training)

        # attention_score = self.dense([att_out, self.kernel, self.bias])
        attention_score = tf.nn.bias_add(tf.tensordot(
            att_out, self.kernel, axes=(-1, 0)), self.bias)

        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(LocalActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttentionSequencePoolingLayer(Layer):
    """The Attentional sequence pooling operation used in DIN.

      Input shape
        - A list of three tensor: [query,keys,keys_length]

        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

        - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **att_activation**: Activation function to use in attention net.

        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

        - **supports_masking**:If True,the input need to support masking.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            if not isinstance(input_shape, list) or len(input_shape) != 3:
                raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
                                 'on a list of 3 inputs')

            if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
                raise ValueError(
                    "Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
                        len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))

            if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
                raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
                                 'inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
                                 'Got different shapes: %s' % (input_shape))
        else:
            pass
        self.local_att = LocalActivationUnit(
            self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, )
        super(AttentionSequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            else:
                tf.print(mask)
                # raise ValueError(
                #     "When supports_masking=False,input must support masking")

            queries, keys = inputs
            tf.print(queries)
            tf.print(keys)
            key_masks = tf.expand_dims(mask[-1], axis=1)

        else:

            queries, keys, keys_length = inputs
            hist_len = keys.get_shape()[1]
            key_masks = tf.sequence_mask(keys_length, hist_len)

        attention_score = self.local_att([queries, keys], training=training)

        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        if tf.__version__ < '1.13.0':
            outputs._uses_learning_phase = attention_score._uses_learning_phase
        else:
            outputs._uses_learning_phase = training is not None

        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[1][1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):

        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None

class DIN(ModelBuilderBase):
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

    def varlen_embedding_lookup(self):
        varlen_embedding_vec_dict = {}

    def concat_func(self,inputs, axis=-1, mask=False):
        if not mask:
            # inputs = list(map(NoMask(), inputs))
            inputs
        if len(inputs) == 1:
            return inputs[0]
        else:
            return tf.keras.layers.Concatenate(axis=axis)(inputs)

    def combined_dnn_input(self,sparse_embedding_list, dense_value_list):
        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            sparse_dnn_input = keras.layers.Flatten()(self.concat_func(sparse_embedding_list))
            dense_dnn_input = keras.layers.Flatten()(self.concat_func(dense_value_list))
            return self.concat_func([sparse_dnn_input, dense_dnn_input])
        elif len(sparse_embedding_list) > 0:
            return keras.layers.Flatten()(self.concat_func(sparse_embedding_list))
        elif len(dense_value_list) > 0:
            return keras.layers.Flatten()(self.concat_func(dense_value_list))
        else:
            raise NotImplementedError("dnn_feature_columns can not be empty list")

    def build(self):
        input_features = self.buildInput()
        history_fc_names = ['hist_item_id', 'hist_cate_id']
        inputs_list = list(input_features.values())
        embDic = self.create_embedding_dict()
        query_emb_list = self.query_embedding_lookup(embDic,input_features)
        keys_emb_list = self.keys_embedding_lookup(embDic, input_features)
        dnn_input_emb_list = self.dnn_input_embedding_lookup(embDic, input_features)
        dense_value_list = self.get_dense_input(input_features)
        keys_emb = self.concat_func(keys_emb_list, mask=True)
        deep_input_emb = self.concat_func(dnn_input_emb_list)
        query_emb = self.concat_func(query_emb_list, mask=True)
        hist = dinUnit((80, 40), 'dice',weight_normalization=False, supports_masking=True)([
            query_emb, keys_emb])
        deep_input_emb = tf.keras.layers.Concatenate()([deep_input_emb, hist])
        # deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
        final_logit_tmp = tf.keras.layers.Dense(1, use_bias=False)(deep_input_emb)
        # dnn_input = self.combined_dnn_input([deep_input_emb], dense_value_list)
        # output = DNN((256, 128, 64), 'relu', 0, 0, False, seed=1024)(dnn_input)
        # final_logit = tf.keras.layers.Dense(1, use_bias=False)(output)
        # output = PredictionLayer('binary')(final_logit)

        model = tf.keras.models.Model(inputs=inputs_list, outputs=final_logit_tmp)
        return model
        # pass

def getData():
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])
    seq_length = np.array([3, 3, 2])  # the actual length of the behavior sequence

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cat_id': cate_id,
                    'his_item_id': hist_iid, 'his_cat_id': hist_cate_id,
                    'pay_score': pay_score, 'seq_length': seq_length}
    x = {name: feature_dict[name] for name in feature_dict.keys()}
    y = np.array([1, 0, 1])
    z = {'input_1': uid, 'input_2': ugender, 'input_3': iid, 'input_4': cate_id,
                    'input_5': hist_iid, 'input_6': hist_cate_id,
                    'input_7': pay_score, 'input_8': seq_length}
    return x,y,z
if __name__=="__main__":
    x,y,z = getData()
    dinModel = DIN("path1","path2")
    buildModel = dinModel.build()
    train = True
    train = False
    if train:
        buildModel.compile('adam', 'binary_crossentropy',
                      metrics=['binary_crossentropy'])
        buildModel.summary()
        history = buildModel.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
        print(buildModel.predict(x))
        # buildModel.save(".\\modelRsTmp")
        tf.keras.models.Model.save(buildModel,".\\modelRsTmp")
    else:
        reloadModel = keras.models.load_model(".\\modelRsTmp")
        print(reloadModel.predict(x))