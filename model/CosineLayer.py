from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import  backend as K
from keras.engine.topology import Layer
import numpy as np

class CosineLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CosineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CosineLayer, self).build(input_shape)

    def call(self, inputs):
        a = K.sum(inputs[0] * inputs[1], axis=1)
        b1 = K.sqrt(K.sum(inputs[0] * inputs[0], axis=1))
        b2 = K.sqrt(K.sum(inputs[1] * inputs[1], axis=1))
        return  a / (b1 * b2)

    def compute_output_shape(self, input_shape):
        return self.output_dim