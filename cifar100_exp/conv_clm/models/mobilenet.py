from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import clm


def mobilenet(inputs,
          net_id,
          emb_size=128,
          is_training=True,
          width_multiplier=1,
          scope='MobileNet'):
  """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn

  inputs = tf.image.resize_images(inputs, [224, 224])
  mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  inputs = inputs-mean
  print('---------------mobilenet--------------------')
  with tf.variable_scope('net_'+str(net_id), reuse=is_training) as sc:
    with slim.arg_scope(
        [slim.convolution2d, slim.separable_convolution2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        biases_initializer=slim.init_ops.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(0.00004)):

        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
          with slim.arg_scope([slim.batch_norm],
                              is_training=is_training,
                              activation_fn=tf.nn.relu,
                              fused=True):
            net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
            print(net)
            net = slim.batch_norm(net, scope='conv_1/batch_norm')
            net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
            net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
            print(net)
            net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
            net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
            print(net)
            net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
            net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')
            print(net)

            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
            net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
            print(net)

            net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
            print(net)
      ################################## conv clm ###################################

  clm_emb = net
  #clm_emb = tf.pad(clm_emb, [[0, 0], [1, 0], [1, 0], [0, 0]])
  clm_emb = clm.clm_enc(clm_emb, net_id, 128, stride=1, padding='SAME', is_training=is_training)
  clm_emb, shared_emb = clm.clm_shared(clm_emb, 128, padding='SAME', is_training=is_training)
  clm_emb = clm.clm_dec(clm_emb, net_id, 1024, stride=1, padding='SAME', is_training=is_training)
  net = clm_emb + net

  with tf.variable_scope('net_'+str(net_id), reuse=is_training) as sc:
    with slim.arg_scope(
        [slim.convolution2d, slim.separable_convolution2d],
        weights_initializer=slim.initializers.xavier_initializer(),
        biases_initializer=slim.init_ops.zeros_initializer(),
        weights_regularizer=slim.l2_regularizer(0.00004)):

        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
          with slim.arg_scope([slim.batch_norm],
                              is_training=is_training,
                              activation_fn=tf.nn.relu,
                              fused=True):

            net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
            net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
            print(net)

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        end_points['squeeze'] = net
        print(net)
        emb = slim.fully_connected(net, emb_size, activation_fn=None, scope='fc_16')
        print(emb)

        end_points['Logits'] = emb
        #predictions = slim.softmax(logits, scope='Predictions')
        #end_points['Predictions'] = predictions

  return emb, shared_emb, None
      #return emb, end_points



'''
mobilenet.default_image_size = 224

def mobilenet_arg_scope(weight_decay=0.0):
  """Defines the default mobilenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
  with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
    return sc
'''
