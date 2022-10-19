import tensorflow as tf
from tensorflow import keras
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
from tensorflow.keras.layers import Layer

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

class dinUnit(Layer):
    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, **kwargs):
        self.hidden_units = att_hidden_units
        self.output_activation = None
        self.activation = 'dice'
        self.seed = 1024
        self.dropout_rate = 0
        self.use_bn = False
        self.l2_reg= 0
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(dinUnit, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def softmax(self,logits, dim=-1, name=None):
        try:
            return tf.nn.softmax(logits, dim=dim, name=name)
        except TypeError:
            return tf.nn.softmax(logits, axis=dim, name=name)
    def build(self, input_shape):
        # self.local_att = LocalActivationUnit(
        #     self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, )
        size = 4 * \
               int(input_shape[0][-1]
                   ) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.atten_kernel = self.add_weight(shape=(size, 1),
                                      initializer=glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.atten_bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")

        #################    dnn - begin ##################################
        input_size = input_shape[-1][-1]
        # hidden_units = [int(input_size)] + list(self.hidden_units)
        hidden_units = [48,80,40]
        self.dnn_kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.dnn_bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [Dice() for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = Dice()

        #################    dnn - end ##################################

        # self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(
        #     x[0], x[1], axes=(-1, 0)), x[2]))

    def call(self, inputs, mask=None, training=None,**kwargs):
        query, keys = inputs
        # tf.print(query)
        # tf.print(keys)
        key_masks = tf.expand_dims(mask[-1], axis=1)


        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = att_input

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                att_out, self.dnn_kernels[i], axes=(-1, 0)), self.dnn_bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            att_out = fc

        attention_score = tf.nn.bias_add(tf.tensordot(
            att_out, self.atten_kernel, axes=(-1, 0)), self.atten_bias)

        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        self.outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            self.outputs = self.softmax(outputs)

        if not self.return_score:
            self.outputs = tf.matmul(outputs, keys)
        return self.outputs