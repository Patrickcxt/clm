# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
import tensorflow as tf
import clm
#from hyper_parameters import *


BN_EPSILON = 0.001

def resnet20_mkb(inputs,
        net_id,
        emb_size=128,
        is_training=True,
        is_mkb_reuse=False,
        scope="Resnet20-mkb"):
    return resnet_mkb(inputs, net_id, emb_size, is_training, is_mkb_reuse, scope, num_layers=20)

def resnet32_mkb(inputs,
        net_id,
        emb_size=128,
        is_training=True,
        is_mkb_reuse=False,
        scope="Resnet32-mkb"):
    return resnet_mkb(inputs, net_id, emb_size, is_training, is_mkb_reuse, scope, num_layers=32)


def resnet_mkb(inputs,
        net_id,
        emb_size,
        is_training,
        is_mkb_reuse,
        scope,
        num_layers):

    def activation_summary(x):
        '''
        :param x: A Tensor
        :return: Add histogram summary and scalar summary of the sparsity of the tensor
        '''
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


    def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        '''
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        '''

        ## TODO: to allow different weight decay to fully connected layer and conv layer
        if is_fc_layer is True:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)

        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables


    def output_layer(input_layer, num_labels):
        '''
        :param input_layer: 2D tensor
        :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
        :return: output layer Y = WX + B
        '''
        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

        fc_h = tf.nn.relu(tf.matmul(input_layer, fc_w) + fc_b)

        return fc_h


    def batch_normalization_layer(input_layer, dimension):
        '''
        Helper function to do batch normalziation
        :param input_layer: 4D tensor
        :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        :return: the 4D tensor after being normalized
        '''
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

        return bn_layer
    """
    def batch_normalization_layer(input_layer, training):
        _BATCH_NORM_DECAY = 0.997
        _BATCH_NORM_EPSILON = 1e-5
        return tf.layers.batch_normalization(
                inputs=input_layer, axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                center=True, scale=True, training=Training, fused=True)
    """


    def conv_bn_relu_layer(input_layer, filter_shape, stride):
        '''
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        out_channel = filter_shape[-1]
        filter = create_variables(name='conv', shape=filter_shape)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = batch_normalization_layer(conv_layer, out_channel)

        output = tf.nn.relu(bn_layer)
        return output


    def bn_relu_conv_layer(input_layer, filter_shape, stride):
        '''
        A helper function to batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''

        in_channel = input_layer.get_shape().as_list()[-1]

        bn_layer = batch_normalization_layer(input_layer, in_channel)
        relu_layer = tf.nn.relu(bn_layer)

        filter = create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer



    def residual_block(input_layer, output_channel, first_block=False):
        '''
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                         input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output


    def inference(input_tensor_batch, net_id, n, reuse, mkb_reuse):
        '''
        The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        :param input_tensor_batch: 4D tensor
        :param n: num_residual_blocks
        :param reuse: To build train graph, reuse=False. To build validation graph and share weights
        with train graph, resue=True
        :return: last layer in the network. Not softmax-ed
        '''

        layers = []
        res = []
        #dataset = "mnist"
        dataset = 'cifar'
        #dataset = 'stl10'
        #with tf.variable_scope('net_'+str(net_id), reuse=reuse):
        with tf.variable_scope('net_'+str(net_id) + '/conv0', reuse=reuse):
            if dataset == 'mnist':
                input_tensor_batch = tf.pad(input_tensor_batch, [[0, 0], [2, 2], [2, 2], [0, 0]])
                input_tensor_batch = tf.cast(input_tensor_batch, tf.float32) / 255.0
                conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 1, 16], 1)
            elif dataset == 'cifar':
                mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                input_tensor_batch = input_tensor_batch - mean
                conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            elif dataset == 'stl10':
                input_tensor_batch = tf.image.resize_images(input_tensor_batch, tf.constant([32, 32]), method=tf.image.ResizeMethod.BILINEAR)
                mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                input_tensor_batch = input_tensor_batch - mean
                conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            #activation_summary(conv0)
            layers.append(conv0)

        print('-------------conv0--------------------')
        print(layers[-1])

        for i in range(n):
            with tf.variable_scope('net_'+str(net_id) + '/conv1_%d' %i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 16, first_block=True)
                else:
                    conv1 = residual_block(layers[-1], 16)
                #activation_summary(conv1)
                layers.append(conv1)

        print('-------------conv1--------------------')
        print(layers[-1])

        ######################MKB 1################################################
        print('mkb_reuse', mkb_reuse)
        with tf.variable_scope('mkb-1', reuse=mkb_reuse):
            mkb1 = clm.clm_shared(layers[-1], 16, padding='SAME')
            mkb1 = mkb1 + layers[-1]
            layers.append(mkb1)
        ######################MKB 1################################################
        res.append(layers[-1])

        for i in range(n):
            with tf.variable_scope('net_'+str(net_id)+'/conv2_%d' %i, reuse=reuse):
                conv2 = residual_block(layers[-1], 32)
                #activation_summary(conv2)
                layers.append(conv2)

        print('-------------conv2--------------------')
        print(layers[-1])

        ######################MKB 2################################################
        with tf.variable_scope('mkb-2', reuse=mkb_reuse):
            mkb2 = clm.clm_shared(layers[-1], 32, padding='SAME')
            mkb2 = mkb2 + layers[-1]
            layers.append(mkb2)
        ######################MKB 2################################################
        res.append(layers[-1])

        for i in range(n):
            with tf.variable_scope('net_'+str(net_id)+'/conv3_%d' %i, reuse=reuse):
                conv3 = residual_block(layers[-1], 64)
                layers.append(conv3)

            assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

        print('-------------conv3--------------------')
        print(layers[-1])

        ######################MKB 3################################################
        with tf.variable_scope('mkb-3', reuse=mkb_reuse):
            mkb3 = clm.clm_shared(layers[-1], 64, padding='SAME')
            mkb3 = mkb3 + layers[-1]
            layers.append(mkb3)
        ######################MKB ################################################
        res.append(layers[-1])


        with tf.variable_scope('net_'+str(net_id), reuse=reuse):
            with tf.variable_scope('fc', reuse=reuse):
                in_channel = layers[-1].get_shape().as_list()[-1]
                bn_layer = batch_normalization_layer(layers[-1], in_channel)
                relu_layer = tf.nn.relu(bn_layer)
                global_pool = tf.reduce_mean(relu_layer, [1, 2])

                #layers.append(global_pool)
                print(global_pool)
                assert global_pool.get_shape().as_list()[-1:] == [64]
                output = output_layer(global_pool, emb_size)
                layers.append(output)

            print('-------------fc--------------------')
            print(layers[-1])

        return layers[-1], res

    if num_layers == 20:
        print('\n==================Resnet 20==============================')
        return inference(inputs, net_id, 3, is_training, is_mkb_reuse)
    elif num_layers == 32:
        print('\n==================Resnet 32==============================')
        return inference(inputs, net_id, 5, is_training, is_mkb_reuse)

"""
def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
"""
