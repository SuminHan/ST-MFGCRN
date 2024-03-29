from tensorflow.keras import backend as K
# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
# from keras.layers import Dense
import tensorflow as tf
import numpy as np


class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # initial_weight_value = np.random.random(input_shape[1:])
        # self.W = K.variable(initial_weight_value)
        self.W = self.add_weight(
            shape=input_shape[1:], initializer="random_normal", trainable=True, dtype=tf.float32, name='W'
        )
        # self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape