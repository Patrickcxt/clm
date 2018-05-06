#! /usr/bin/env python
"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Association-based semi-supervised training example in MNIST dataset.

Training should reach ~1% error rate on the test set using 100 labeled samples
in 5000-10000 steps (a few minutes on Titan X GPU)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import backend
import architectures
import mobilenet
import resnet

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 10,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 10,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('sup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.8, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 1000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 100000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

from tools import cifar as cifar_tools
from conviz import conviz
from data_layer import data_layer

NUM_LABELS = cifar_tools.NUM_LABELS
IMAGE_SHAPE = cifar_tools.IMAGE_SHAPE

def valid_and_test(model, images, labels):
    (test_pred_0, test_pred_1) = model.classify(images)
    test_pred_0 = test_pred_0.argmax(-1)
    test_pred_1 = test_pred_1.argmax(-1)
    conf_mtx_0 = backend.confusion_matrix(labels, test_pred_0, NUM_LABELS)
    conf_mtx_1 = backend.confusion_matrix(labels, test_pred_1, NUM_LABELS)
    test_err_0 = (labels != test_pred_0).mean() * 100
    test_err_1 = (labels != test_pred_1).mean() * 100
    print(conf_mtx_0)
    print(conf_mtx_1)
    return test_err_0, test_err_1

def main(_):

  dl = data_layer('cifar10')
  train_images, train_labels = dl.get_train_set()
  valid_images, valid_labels = dl.get_valid_set()
  test_images, test_labels = dl.get_test_set()

  graph = tf.Graph()
  with graph.as_default():
    model_func = [resnet.resnet32, architectures.mnist_model]
    model = backend.SemisupModel(model_func, NUM_LABELS, IMAGE_SHAPE)

    # Set up inputs.

    sup_images_0 = tf.placeholder(tf.float32, [FLAGS.sup_batch_size] + IMAGE_SHAPE)
    sup_labels_0 = tf.placeholder(tf.int32, [FLAGS.sup_batch_size])
    sup_images_1= tf.placeholder(tf.float32, [FLAGS.sup_batch_size] + IMAGE_SHAPE)
    sup_labels_1 = tf.placeholder(tf.int32, [FLAGS.sup_batch_size])

    # Compute embeddings and logits.
    sup_emb_0, sup_shared_0, _  = model.image_to_embedding(0, sup_images_0)
    sup_logit_0 = model.embedding_to_logit(0, sup_emb_0)

    sup_emb_1, sup_shared_1, _ = model.image_to_embedding(1, sup_images_1)
    sup_logit_1 = model.embedding_to_logit(1, sup_emb_1)

    # Add losses after final prediction.
    model.add_logit_loss(0, sup_logit_0, sup_labels_0)
    model.add_logit_loss(1, sup_logit_1, sup_labels_1)
    model.add_kl_loss(0, sup_logit_1, sup_logit_0, weight=0.01)
    model.add_kl_loss(1, sup_logit_0, sup_logit_1, weight=0.01)

    # Add losses after between clm
    sup_shared_0 = slim.flatten(sup_shared_0, scope='kl_flatten_0')
    #sup_shared_0 = model.shared_to_embedding(0, sup_shared_0, 128, is_training=False)
    sup_shared_0 = tf.nn.l2_normalize(sup_shared_0, 1, 1e-10, name='shared_emb_0')
    sup_shared_1 = slim.flatten(sup_shared_1, scope='kl_flatten_1')
    #sup_shared_1 = model.shared_to_embedding(1, sup_shared_1, 128, is_training=False)
    sup_shared_1 = tf.nn.l2_normalize(sup_shared_1, 1, 1e-10, name='shared_emb_1')

    model.add_cosine_loss(0, sup_shared_0, sup_shared_1, weight=1)
    model.add_cosine_loss(1, sup_shared_1, sup_shared_0, weight=1)


    t_learning_rate_0 = tf.train.exponential_decay(
        1e-1,
        model.step[0],
        30000,
        0.1,
        staircase=True)
    t_learning_rate_1 = tf.train.exponential_decay(
        1e-2,
        model.step[1],
        30000,
        0.1,
        staircase=True)
    t_learning_rate = [t_learning_rate_0, t_learning_rate_1]
    train_op = model.create_separate_train_op(t_learning_rate,'kl')

    #summary_op = tf.summary.merge_all()
    print(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op_0 = tf.summary.merge([s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if 'Net_0' in s.name])
    summary_op_1 = tf.summary.merge([s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if 'Net_1' in s.name])

    summary_writer = tf.summary.FileWriter(FLAGS.logdir+'/summaries/', graph)

    saver = tf.train.Saver()

  gpuConfig = tf.ConfigProto()
  gpuConfig.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config=gpuConfig) as sess:
    tf.global_variables_initializer().run()

    #saver.restore(sess, './l1_loss/2_1/models/model-50000')

    min_err_0, min_err_1 = 100, 100

    # Training

    for step in xrange(FLAGS.max_steps):

      images_0, labels_0 = dl.get_next_batch(0)
      _, total_losses_0, logit_losses_0, kl_losses_0, cosine_losses_0, summaries_0 = sess.run([train_op[0], model.train_loss[0], model.logit_loss[0],  model.kl_loss[0], model.cosine_loss[0], summary_op_0],
                                                      feed_dict={sup_images_0: images_0,
                                                                 sup_labels_0: labels_0,
                                                                 sup_images_1: images_0,
                                                                 sup_labels_1: labels_0})

      images_1, labels_1 = dl.get_next_batch(1)
      _, total_losses_1, logit_losses_1,  kl_losses_1, cosine_losses_1, summaries_1 = sess.run([train_op[1], model.train_loss[1], model.logit_loss[1], model.kl_loss[1], model.cosine_loss[1], summary_op_1],
                                                      feed_dict={sup_images_1: images_1,
                                                                 sup_labels_1: labels_1,
                                                                 sup_images_0: images_1,
                                                                 sup_labels_0: labels_1})

      """
      print('Step: ', step)
      print('resnet loss: ' + str(total_losses_0) +  '  small loss: ' + str(total_losses_1))
      print('logit_loss: ', logit_losses_0, logit_losses_1)
      print('mse_loss: ' + str(mse_losses_0) + ' ' + str(mse_losses_1))
      """
      #print('kl_loss: ', kl_losses_0, kl_losses_1)
      if step % 50 == 0:
        print('Step: ', step)
        print('resnet loss: ' + str(total_losses_0) +  '  small loss: ' + str(total_losses_1))
        print('logit_loss: ' + str(logit_losses_0) + ' ' + str(logit_losses_1))
        print('cosine_loss: ', cosine_losses_0, cosine_losses_1)
        #print('mse_loss: ' + str(mse_losses_0) + ' ' + str(mse_losses_1))
        print('kl_loss: ', kl_losses_0, kl_losses_1)
        #print('visit_loss: ' + str(visit_losses[0]) + ' ' + str(visit_losses[1]))
        print()

        """
        (test_pred_0, test_pred_1) = model.classify(images_1)
        test_pred_0 = test_pred_0.argmax(-1)
        test_pred_1 = test_pred_1.argmax(-1)
        conf_mtx_0 = backend.confusion_matrix(labels_1, test_pred_0, NUM_LABELS)
        conf_mtx_1 = backend.confusion_matrix(labels_1, test_pred_1, NUM_LABELS)
        test_err_0 = (labels_1 != test_pred_0).mean() * 100
        test_err_1 = (labels_1 != test_pred_1).mean() * 100
        print('Train error resnet: %.2f %%' % test_err_0)
        print('Train error small: %.2f %%' % test_err_1)

        train_summary_0 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Train Err resnet', simple_value=test_err_0)])

        train_summary_1 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Train Err smallnet', simple_value=test_err_1)])
        #summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(train_summary_0, step)
        summary_writer.add_summary(train_summary_1, step)
        """

        # plot conv output of clm
      """
      if step % 500 == 0:
        root_dir = 'out/alt-kl-l2/'
        conviz.plot_conv_output(conv_out[0], root_dir, 'step{}'.format(step), '1_in')
        conviz.plot_conv_output(conv_out[1], root_dir, 'step{}'.format(step), '2_enc')
        conviz.plot_conv_output(conv_out[2], root_dir, 'step{}'.format(step), '3_emb')
        conviz.plot_conv_output(conv_out[3], root_dir, 'step{}'.format(step), '4_clm')
        conviz.plot_conv_output(conv_out[4], root_dir, 'step{}'.format(step), '5_dec')
        conviz.plot_conv_output(conv_out[5], root_dir, 'step{}'.format(step), '6_out')
      """

      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
      #if step == 0:
        print('Step: %d' % step)
        valid_err_0, valid_err_1 = valid_and_test(model, valid_images, valid_labels)
        min_err_0 = min(valid_err_0, min_err_0)
        min_err_1 = min(valid_err_1, min_err_1)
        print('Valid error resnet: %.2f %%' % valid_err_0)
        print('Valid error small: %.2f %%' % valid_err_1)
        print('Min Valid error: %.2f %%, %.2f%%' % (min_err_0, min_err_1))

        print()

        valid_summary_0 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Valid Err resnet', simple_value=valid_err_0)])

        valid_summary_1 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Valid Err smallnet', simple_value=valid_err_1)])
        #summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(summaries_0, step)
        summary_writer.add_summary(summaries_1, step)
        summary_writer.add_summary(valid_summary_0, step)
        summary_writer.add_summary(valid_summary_1, step)

        saver.save(sess, FLAGS.logdir+'/models/model', model.step[0])
    print('\n\n==========================Test Error===============================')
    test_err_0, test_err_1 = valid_and_test(model, test_images, test_labels)
    print('Test error resnet: %.2f %%' % test_err_0)
    print('Test error small: %.2f %%' % test_err_1)




if __name__ == '__main__':
  app.run()
