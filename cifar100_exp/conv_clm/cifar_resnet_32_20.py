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

NUM_LABELS = cifar_tools.NUM_LABELS_100
IMAGE_SHAPE = cifar_tools.IMAGE_SHAPE

def main(_):

  dl = data_layer('cifar100')
  train_images, train_labels = dl.get_train_set()
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
    sup_emb_0, sup_shared_0, conv_out_0  = model.image_to_embedding(0, sup_images_0)
    sup_logit_0 = model.embedding_to_logit(0, sup_emb_0)

    sup_emb_1, sup_shared_1, conv_out_1 = model.image_to_embedding(1, sup_images_1)
    sup_logit_1 = model.embedding_to_logit(1, sup_emb_1)

    # Add losses after final prediction.
    model.add_logit_loss(0, sup_logit_0, sup_labels_0)
    model.add_logit_loss(1, sup_logit_1, sup_labels_1)
    model.add_kl_loss(0, sup_logit_1, sup_logit_0, weight=0.01)
    model.add_kl_loss(1, sup_logit_0, sup_logit_1, weight=0.01)

    # Add losses after between clm
    sup_shared_0 = slim.flatten(sup_shared_0, scope='kl_flatten_0')
    #sup_shared_0 = model.shared_to_embedding(sup_shared_0, 128, is_training=False)
    sup_shared_0 = tf.nn.l2_normalize(sup_shared_0, 1, 1e-10, name='shared_emb_0')
    sup_shared_1 = slim.flatten(sup_shared_1, scope='kl_flatten_1')
    #sup_shared_1 = model.shared_to_embedding(sup_shared_1, 128, is_training=True)
    sup_shared_1 = tf.nn.l2_normalize(sup_shared_1, 1, 1e-10, name='shared_emb_1')

    print('------------sup_shared_emb------------------')
    print(sup_shared_0)
    print(sup_shared_1)

    model.add_cosine_loss(0, sup_shared_0, sup_shared_1, weight=1)
    model.add_cosine_loss(1, sup_shared_1, sup_shared_0, weight=1)


    t_learning_rate = tf.train.exponential_decay(
        1e-2,
        model.step[0],
        30000,
        0.1,
        staircase=True)
    train_op = model.create_train_op_type(t_learning_rate,'kl')
    #train_op = model.create_train_op(t_learning_rate)


    summary_op = tf.summary.merge_all()

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
      _, total_losses, logit_losses, cosine_losses, kl_losses, summaries, conv_out = sess.run([train_op, model.train_loss, model.logit_loss, model.cosine_loss, model.kl_loss, summary_op, conv_out_1],
                                                      feed_dict={sup_images_0: images_0,
                                                                 sup_labels_0: labels_0,
                                                                 sup_images_1: images_0,
                                                                 sup_labels_1: labels_0})

      """
      print('Step: ', step)
      print('total_loss: ', total_losses)
      print('logit_loss: ', logit_losses)
      print('cosine_loss: ', cosine_losses)
      #print('mse_loss: ', mse_losses)
      #print('kl_loss: ', kl_losses)
      """
      if step % 100 == 0:
        print('Step: ', step)
        print('total_loss: ', total_losses)
        print('logit_loss: ', logit_losses)
        print('cosine_loss: ', cosine_losses)
        print('kl_loss: ', kl_losses)
        #print('mse_loss: ', mse_losses)
        #print('kl_loss: ' + str(kl_losses_0) + ' ' + str(kl_losses_1))
        #print('visit_loss: ' + str(visit_losses[0]) + ' ' + str(visit_losses[1]))
        print()

        (test_pred_0, test_pred_1) = model.classify(images_0)
        #test_pred_0, test_pred_1 = shared_0, shared_1
        test_pred_0 = test_pred_0.argmax(-1)
        test_pred_1 = test_pred_1.argmax(-1)
        conf_mtx_0 = backend.confusion_matrix(labels_0, test_pred_0, NUM_LABELS)
        conf_mtx_1 = backend.confusion_matrix(labels_0, test_pred_1, NUM_LABELS)
        test_err_0 = (labels_0 != test_pred_0).mean() * 100
        test_err_1 = (labels_0 != test_pred_1).mean() * 100
        print('Train error resnet: %.2f %%' % test_err_0)
        print('Train error small: %.2f %%' % test_err_1)

        train_summary_0 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Train Err resnet', simple_value=test_err_0)])

        train_summary_1 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Train Err smallnet', simple_value=test_err_1)])
        summary_writer.add_summary(train_summary_0, step)
        summary_writer.add_summary(train_summary_1, step)

        # plot conv output of clm
      """
      if step % 500 == 0:
        root_dir = 'out/syn/'
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
        (test_pred_0, test_pred_1) = model.classify(test_images)
        test_pred_0 = test_pred_0.argmax(-1)
        test_pred_1 = test_pred_1.argmax(-1)
        conf_mtx_0 = backend.confusion_matrix(test_labels, test_pred_0, NUM_LABELS)
        conf_mtx_1 = backend.confusion_matrix(test_labels, test_pred_1, NUM_LABELS)
        test_err_0 = (test_labels != test_pred_0).mean() * 100
        test_err_1 = (test_labels != test_pred_1).mean() * 100
        min_err_0 = min(test_err_0, min_err_0)
        min_err_1 = min(test_err_1, min_err_1)
        print(conf_mtx_0)
        print(conf_mtx_1)
        print('Test error resnet: %.2f %%' % test_err_0)
        print('Test error small: %.2f %%' % test_err_1)
        print('Min Test error: %.2f %%, %.2f%%' % (min_err_0, min_err_1))

        print()

        test_summary_0 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test Err resnet', simple_value=test_err_0)])

        test_summary_1 = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test Err smallnet', simple_value=test_err_1)])
        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(test_summary_0, step)
        summary_writer.add_summary(test_summary_1, step)

        saver.save(sess, FLAGS.logdir+'/models/model', model.step[0])



if __name__ == '__main__':
  app.run()
