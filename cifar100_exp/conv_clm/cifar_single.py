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
#import architectures
#import mobilenet
#import resnet
import backend

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


flags.DEFINE_integer('sup_batch_size', 128,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

#flags.DEFINE_float('learning_rate', 0.5e-1, 'Initial learning rate.')

#flags.DEFINE_float('decay_factor', 0.1, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 30000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_integer('max_steps', 64000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

flags.DEFINE_string('subset', 'train', 'Training or test.')

from tools import cifar as cifar_tools
from data_layer import data_layer
#from models import resnet_old as resnet
#from models import resnet as resnet
from models import resnet_mkb as resnet_mkb

NUM_LABELS = cifar_tools.NUM_LABELS
IMAGE_SHAPE = cifar_tools.IMAGE_SHAPE

def valid_and_test(model, images, labels):
  test_pred_0 = model.classify_single(images)
  test_pred_0 = test_pred_0.argmax(-1)
  conf_mtx_0 = backend.confusion_matrix(labels, test_pred_0, NUM_LABELS)
  test_err_0 = (labels != test_pred_0).mean() * 100
  print(conf_mtx_0)
  return test_err_0

def main(_):

  dl = data_layer('cifar10', FLAGS.sup_batch_size)
  train_images, train_labels = dl.get_train_set()
  valid_images, valid_labels = dl.get_valid_set()
  test_images, test_labels = dl.get_test_set()

  # Sample labeled training subset.

  graph = tf.Graph()
  with graph.as_default():
    #model_func = [resnet.resnet32]
    model_func = [resnet_mkb.resnet32_mkb]
    model = backend.SemisupModel(model_func, NUM_LABELS, IMAGE_SHAPE)

    # Set up inputs.

    sup_images_0 = tf.placeholder(tf.float32, [None] + IMAGE_SHAPE)
    sup_labels_0 = tf.placeholder(tf.int32, [None])

    # Compute embeddings and logits.
    sup_emb_0, _ = model.image_to_embedding(0, sup_images_0)
    sup_logit_0 = model.embedding_to_logit(0, sup_emb_0)

    # Add losses.
    model.add_logit_loss(0, sup_logit_0, sup_labels_0)

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
        [32000, 48000],
        [1e-1, 1e-2, 1e-3])
    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir+'/summaries/', graph)

    saver = tf.train.Saver(max_to_keep=10)

  gpuConfig = tf.ConfigProto()
  gpuConfig.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config=gpuConfig) as sess:
    tf.global_variables_initializer().run()

    # Load vgg weights
    #model.load_vgg_weights(0, sess)

    if FLAGS.subset == 'train':
        min_err_0 = 100.0
        for step in xrange(FLAGS.max_steps):
          images, labels = dl.get_next_batch(0)
          #images, labels = dl.get_batch_per_class(FLAGS.sup_per_class)
          _, train_loss, logit_loss, summaries = sess.run([train_op, model.train_loss, model.logit_loss[0][0], summary_op], feed_dict={sup_images_0: images, sup_labels_0: labels})

          #print('Step: %d' % step)
          #print(train_loss, ' ', logit_loss)
          if (step) % 200 == 0:
            print('Step: %d' % step)
            print(train_loss, ' ', logit_loss)
          if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
            print('Step: %d' % step)
            print(train_loss, ' ', logit_loss)
            valid_err_0 = valid_and_test(model, valid_images, valid_labels)
            min_err_0 = min(valid_err_0, min_err_0)
            print('Valid error: %.2f %%' % valid_err_0)
            print('Min Valid error: %.2f %%' % min_err_0)
            print()

            valid_summary_0 = tf.Summary(
                value=[tf.Summary.Value(
                    tag='Valid Err', simple_value=valid_err_0)])

            summary_writer.add_summary(summaries, step)
            summary_writer.add_summary(valid_summary_0, step)

          if (step+1) % 10000 == 0:
            saver.save(sess, FLAGS.logdir+'/models/model', model.step[0])

        print('=========================Test Error+==============================')
        test_err = valid_and_test(model, test_images, test_labels)
        print('Final test error: %.2f %%' % test_err)
    else:
        saver.restore(sess, './save_cifar100/r20/models/-50000')
        print('=========================Test Error+==============================')
        test_err = valid_and_test(model, test_images, test_labels)
        print('Final test error: %.2f %%' % test_err)

if __name__ == '__main__':
  app.run()
