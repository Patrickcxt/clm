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

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-1, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.1, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 30000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_integer('max_steps', 80000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

flags.DEFINE_string('subset', 'train', 'Training or test.')

from tools import cifar as cifar_tools
#from conviz import conviz
from data_layer import data_layer
from models import resnet_mkb as resnet_mkb
#from models import resnet_mkb_old as resnet_mkb

NUM_LABELS = cifar_tools.NUM_LABELS
IMAGE_SHAPE = cifar_tools.IMAGE_SHAPE

def valid_and_test(model, images, labels):
    (pred_0, pred_1) = model.classify(images)
    pred_0 = pred_0.argmax(-1)
    pred_1 = pred_1.argmax(-1)
    conf_mtx_0 = backend.confusion_matrix(labels, pred_0, NUM_LABELS)
    conf_mtx_1 = backend.confusion_matrix(labels, pred_1, NUM_LABELS)
    err_0 = (labels != pred_0).mean() * 100
    err_1 = (labels != pred_1).mean() * 100
    print(conf_mtx_0)
    print(conf_mtx_1)
    return err_0, err_1

def main(_):

  dl = data_layer('cifar10', FLAGS.sup_batch_size)
  train_images, train_labels = dl.get_train_set()
  valid_images, valid_labels = dl.get_valid_set()
  test_images, test_labels = dl.get_test_set()

  graph = tf.Graph()
  with graph.as_default():
    model_func = [resnet_mkb.resnet32_mkb, resnet_mkb.resnet20_mkb]
    model = backend.SemisupModel(model_func, NUM_LABELS, IMAGE_SHAPE)

    # Set up inputs.

    sup_images_0 = tf.placeholder(tf.float32, [FLAGS.sup_batch_size] + IMAGE_SHAPE)
    sup_labels_0 = tf.placeholder(tf.int32, [FLAGS.sup_batch_size])
    sup_images_1= tf.placeholder(tf.float32, [FLAGS.sup_batch_size] + IMAGE_SHAPE)
    sup_labels_1 = tf.placeholder(tf.int32, [FLAGS.sup_batch_size])

    # Compute embeddings and logits.
    sup_emb_0, sup_res_0  = model.image_to_embedding(0, sup_images_0)
    sup_logit_0 = model.embedding_to_logit(0, sup_emb_0)

    sup_emb_1, sup_res_1 = model.image_to_embedding(1, sup_images_1)
    sup_logit_1 = model.embedding_to_logit(1, sup_emb_1)

    # Add losses after final prediction.
    model.add_logit_loss(0, sup_logit_0, sup_labels_0)
    model.add_logit_loss(1, sup_logit_1, sup_labels_1)
    model.add_kl_loss(0, sup_logit_1, sup_logit_0, weight=0.01)
    model.add_kl_loss(1, sup_logit_0, sup_logit_1, weight=0.01)

    # Add losses after between clm
    """
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
    """


    """
    t_learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        model.step[0],
        FLAGS.decay_steps,
        FLAGS.decay_factor,
        staircase=True)
    """
    t_learning_rate = tf.train.piecewise_constant(
        model.step[0],
        [48000, 64000],
        [1e-1, 1e-2, 1e-3])
    train_op = model.create_train_op_type(t_learning_rate,'kl')
    #train_op = model.create_train_op(t_learning_rate)


    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir+'/summaries/', graph)

    saver = tf.train.Saver(max_to_keep=10)

  gpuConfig = tf.ConfigProto()
  gpuConfig.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config=gpuConfig) as sess:
    tf.global_variables_initializer().run()


    if FLAGS.subset == 'train':
        min_err_0, min_err_1 = 100, 100

        # Training

        for step in xrange(FLAGS.max_steps):

          images_0, labels_0 = dl.get_next_batch(0)
          _, total_losses, logit_losses,  kl_losses, summaries = sess.run([train_op, model.train_loss, model.logit_loss,  model.kl_loss, summary_op],
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
            #print('cosine_loss: ', cosine_losses)
            print('kl_loss: ', kl_losses)
            #print('mse_loss: ', mse_losses)
            #print('kl_loss: ' + str(kl_losses_0) + ' ' + str(kl_losses_1))
            #print('visit_loss: ' + str(visit_losses[0]) + ' ' + str(visit_losses[1]))
            print()

            train_err_0, train_err_1 = valid_and_test(model, images_0, labels_0)
            print('Train error resnet: %.2f %%' % train_err_0)
            print('Train error small: %.2f %%' % train_err_1)

            train_summary_0 = tf.Summary(
                value=[tf.Summary.Value(
                    tag='Train Err resnet32', simple_value=train_err_0)])

            train_summary_1 = tf.Summary(
                value=[tf.Summary.Value(
                    tag='Train Err resnet20', simple_value=train_err_1)])
            summary_writer.add_summary(train_summary_0, step)
            summary_writer.add_summary(train_summary_1, step)


          if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
          #if step == 0:
            print('Step: %d' % step)
            valid_err_0, valid_err_1 = valid_and_test(model, valid_images, valid_labels)
            min_err_0 = min(valid_err_0, min_err_0)
            min_err_1 = min(valid_err_1, min_err_1)
            print('Valid error resnet32: %.2f %%' % valid_err_0)
            print('Valid error resnet20: %.2f %%' % valid_err_1)
            print('Min Valid error: %.2f %%, %.2f%%' % (min_err_0, min_err_1))

            print()

            valid_summary_0 = tf.Summary(
                value=[tf.Summary.Value(
                    tag='Valid Err resnet32', simple_value=valid_err_0)])

            valid_summary_1 = tf.Summary(
                value=[tf.Summary.Value(
                    tag='Valid Err resnet20', simple_value=valid_err_1)])
            summary_writer.add_summary(summaries, step)
            summary_writer.add_summary(valid_summary_0, step)
            summary_writer.add_summary(valid_summary_1, step)

        if (step+1) % 10000:
            saver.save(sess, FLAGS.logdir+'/models/model', model.step[0])

        print('=====================Test Error===============================')
        test_err_0, test_err_1 = valid_and_test(model, test_images, test_labels)
        print('Test error: %.2f %%, %.2f%%' % (test_err_0, test_err_1))
    else:
        #saver.restore(sess, './save_newbn/r32_r20/models/model-100000')
        saver.restore(sess, './save_cifar100/r32_r20/models/model-48000')
        print('=====================Test Error===============================')
        test_err_0, test_err_1 = valid_and_test(model, test_images, test_labels)
        print('Test error: %.2f %%, %.2f%%' % (test_err_0, test_err_1))



if __name__ == '__main__':
  app.run()
