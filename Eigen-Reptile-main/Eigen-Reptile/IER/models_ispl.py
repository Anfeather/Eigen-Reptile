"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)


# pylint: disable=R0903
# class OmniglotModel:
#     """
#     A model for Omniglot classification.
#     """
#     def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs ):
#         self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
#         out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
#         for _ in range(4):
#             out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
#             out = tf.layers.batch_normalization(out, training=True)
#             out = tf.nn.relu(out)


#         out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
#         self.logits = tf.layers.dense(out, num_classes)

        

#         self.label_ph = tf.placeholder(tf.int32, shape=(None,))
#         self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
#                                                                    logits=self.logits)

#         # self.pace_param = tf.constant(0.00000001)                                                           
#         self.lp = tf.Variable(4.5)          #  initialize learning_pace lp >0
#         self.lp = tf.stop_gradient(self.lp)
#         # self.s = self.loss.get_shape()
#         self.v = tf.cast(tf.less(self.loss, self.lp,name=None),dtype=tf.float32)
#         self.v = tf.stop_gradient(self.v)
#         self.losses = tf.subtract(tf.reduce_sum(self.v * self.loss), self.lp * tf.reduce_sum(self.v),name=None)
#         # self.lp += self.pace_param

        
#         self.predictions = tf.argmax(self.logits, axis=-1)
#         self.minimize_op = optimizer(**optim_kwargs).minimize(self.losses)









# pylint: disable=R0903
class MiniImageNetModel:
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32 , 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)

#         # self.pace_param = tf.constant(0.00000001)                                                           
#         self.lp = tf.Variable(4.5)          #  initialize learning_pace lp >0
#         self.lp = tf.stop_gradient(self.lp)
#         # self.s = self.loss.get_shape()
#         self.v = tf.cast(tf.less(self.loss, self.lp,name=None),dtype=tf.float32)
#         self.v = tf.stop_gradient(self.v)
#         self.losses = tf.subtract(tf.reduce_sum(self.v * self.loss), self.lp * tf.reduce_sum(self.v),name=None)
#         # self.lp += self.pace_param
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)




class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)