from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
a = [1, 2]
b = tf.convert_to_tensor(a)
print(b)
print(b[1])