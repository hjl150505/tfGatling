import tensorflow as tf

import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Layer
import sys

# tf.enable_eager_execution()

class Dice(Layer):

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name='dice_alpha')
        super(Dice, self).build(input_shape)
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class dinUnit(Layer):
    def __init__(self, att_hidden_units=(32, 16), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, **kwargs):
        self.hidden_units = att_hidden_units
        self.output_activation = None
        self.activation = 'dice'
        self.seed = 1024
        self.dropout_rate = 0.2
        self.use_bn = False
        self.l2_reg = 0.01
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(dinUnit, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def softmax(self, logits, dim=-1, name=None):
        try:
            return tf.nn.softmax(logits, dim=dim, name=name)
        except TypeError:
            return tf.nn.softmax(logits, axis=dim, name=name)

    def build(self, input_shape):
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
        hidden_units = [80,32,16]
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

    def call(self, inputs, mask=None, training=None, **kwargs):
        keys,query  = inputs
        key_masks = tf.expand_dims(mask[-1], axis=1)
        if len(keys.get_shape())==4:
            keys = tf.squeeze(input=keys,axis=[1])
        # print(f'keys_value:{keys}')
        # tf.print(keys,output_stream = sys.stderr)
        # print(keys.eval())
        # print(f'keys_type:{type(keys)}')
        keys_len = keys.get_shape()[1]
        # print(f'keys_len:{keys_len}')
        query_len = query.get_shape()[1]
        # print(f'query_len:{query_len}')
        queries = K.repeat_elements(query, keys_len, 1)
        # tf.print("queries.shape"*20)
        # print("queries.shape"*20)
        # print(f'queries_shape:{queries.get_shape()}')
        # tf.print(queries)
        if len(keys.get_shape())==4:
            keys = tf.squeeze(input=keys,axis=[1])
        # print(f'keys_shape:{keys.get_shape()}')

        shapeSub = queries-keys
        if len(shapeSub.get_shape())==4:
            shapeSub = tf.squeeze(input=queries-keys,axis=[1])

        # print(f'shapeSub_shape:{shapeSub.get_shape()}')

        shapeMult =  queries * keys

        if len(shapeMult.get_shape()) == 4:
            shapeMult = tf.squeeze(input=queries * keys, axis=[1])

        # print(len(shapeMult.get_shape()))
        # print(f'shapeMult_shape:{shapeMult.get_shape()}')

        # att_input = tf.concat(
        #     [queries, keys, queries - keys, queries * keys], axis=-1)
        # print(f'queirs_shape_2:{queries.get_shape()}')
        # print(f'keys_shape_2:{keys.get_shape()}')
        # print(f'shapeSub_shape_2:{shapeSub.get_shape()}')
        # print(f'shapeMult_shape_2:{shapeMult.get_shape()}')
        att_input = tf.concat(
            [queries, keys, shapeSub, shapeMult], axis=-1)

        att_out = att_input
        # print(f'att_out_shape:{att_out.get_shape()}')
        for i in range(len(self.hidden_units)):
            # print(f'i_data:{i}')
            # print(f'self.dnn_kernels[i]_shape:{self.dnn_kernels[i].get_shape()}')
            # print(f'self.dnn_bias[i]_shape:{self.dnn_bias[i].get_shape()}')
            test_data = tf.tensordot(
                att_out, self.dnn_kernels[i], axes=(-1, 0))
            # print(f'test_data_shape:{test_data.get_shape()}')
            fc = tf.nn.bias_add(test_data, self.dnn_bias[i])

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
