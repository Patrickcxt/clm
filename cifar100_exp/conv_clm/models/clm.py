import os
import re

import tensorflow as tf
#import tensorlayer as tl
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
from scipy.misc import imread, imresize
from scipy.misc.pilutil import imshow


#from tfutils import w, b

def batch_normalization_layer(input_layer, is_training):
    _BATCH_NORM_DECAY = 0.997
    _BATCH_NORM_EPSILON = 1e-3
    return tf.layers.batch_normalization(
        inputs=input_layer, axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        center=True, scale=True, training=is_training, fused=True)

def encoder(inputs, out_channel, stride, padding, l2_weight=1e-3):
    with slim.arg_scope([slim.conv2d],
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        preds = slim.conv2d(inputs, out_channel, [3, 3], stride=stride, scope='conv1', padding=padding)
        in_channel = preds.get_shape().as_list()[-1]
        preds = batch_normalization_layer(preds, in_channel)
        preds = tf.nn.relu(preds)
    return preds

def decoder(inputs, out_channel, out_shape, stride, padding, l2_weight=1e-3):
    """
    with slim.arg_scope([slim.conv2d_transpose],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
    """
    de_weight = tf.get_variable('de_weight', shape=[3, 3, out_channel[1], out_channel[0]])
    de_bias = tf.get_variable('de_bias', shape=[out_channel[1]])
    preds = tf.nn.conv2d_transpose(inputs, de_weight, out_shape, strides=stride, padding=padding) + de_bias
    #preds = tf.nn.sigmoid(preds)
    preds = tf.nn.elu(preds)
    return preds

def clm_enc(inputs, net_id, out_channel, stride=[1, 1, 1, 1], padding='SAME', is_training=True):
    with tf.variable_scope('net_' + str(net_id) + '/encoder', reuse=is_training):
        return encoder(inputs, out_channel, stride, padding)

def clm_dec(inputs, net_id, out_channel,  stride=[1, 1, 1, 1], padding='SAME', is_training=True):
    with tf.variable_scope('net_' + str(net_id) + '/decoder', reuse=is_training):
        #return decoder(inputs, out_channel, out_shape, stride, padding)
        return encoder(inputs, out_channel, stride, padding)

def clm_shared(inputs, out_channel, padding='SAME', l2_weight=1e-3, reuse=False, is_train=True):
    with slim.arg_scope([slim.conv2d],
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
            preds = inputs
            with tf.variable_scope('clm_layer_1', reuse=reuse):
                preds = slim.conv2d(preds, out_channel, [3, 3], scope='conv1', padding=padding)
                #preds = batch_normalization_layer(preds, is_training=is_train)
                preds = tf.nn.relu(preds)

            with tf.variable_scope('clm_layer_2', reuse=reuse):
                preds = slim.conv2d(preds, out_channel, [3, 3], scope='conv2', padding=padding)
                #preds = batch_normalization_layer(preds, is_training=is_train)
                preds = tf.nn.relu(preds)
    return preds

