from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class ThresholdedBinary(Layer):
    def __init__(self,  theta=0.5, **kwargs):
        super(ThresholdedBinary, self).__init__(**kwargs)
        # self.supports_masking = True
        self.theta = K.cast_to_floatx(theta)

    def call(self, inputs):
        temp = K.ones_like(inputs)
        print(type(inputs))
        print(type(temp))
        return temp * K.cast(K.greater(inputs, self.theta), K.floatx())
        # return inputs * K.cast(K.greater(inputs, self.theta), K.floatx())

    # def get_config(self):
    #     config = {"theta" : float(self.theta)}
    #     base_config = super(ThresholdedBinary, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))