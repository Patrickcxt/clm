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

Utility functions for Association-based semisupervised training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim


def create_input(input_images, input_labels, batch_size):
  """Create preloaded data batch inputs.

  Args:
    input_images: 4D numpy array of input images.
    input_labels: 2D numpy array of labels.
    batch_size: Size of batches that will be produced.

  Returns:
    A list containing the images and labels batches.
  """
  if input_labels is not None:
    image, label = tf.train.slice_input_producer([input_images, input_labels])
    return tf.train.batch([image, label], batch_size=batch_size)
  else:
    image = tf.train.slice_input_producer([input_images])
    return tf.train.batch(image, batch_size=batch_size)


def create_per_class_inputs(image_by_class, n_per_class, class_labels=None):
  """Create batch inputs with specified number of samples per class.

  Args:
    image_by_class: List of image arrays, where image_by_class[i] containts
        images sampled from the class class_labels[i].
    n_per_class: Number of samples per class in the output batch.
    class_labels: List of class labels. Equals to range(len(image_by_class)) if
        not provided.

  Returns:
    images: Tensor of n_per_class*len(image_by_class) images.
    labels: Tensor of same number of labels.
  """
  if class_labels is None:
    class_labels = np.arange(len(image_by_class))  # [0, 1, 2, ..., 9]
  batch_images, batch_labels = [], []
  for images, label in zip(image_by_class, class_labels):
    labels = tf.fill([len(images)], label)
    images, labels = create_input(images, labels, n_per_class)
    batch_images.append(images)
    batch_labels.append(labels)
  return tf.concat(batch_images, 0), tf.concat(batch_labels, 0)


def sample_by_label(images, labels, n_per_label, num_labels, seed=None):
  """Extract equal number of sampels per class."""
  res = []
  rng = np.random.RandomState(seed=seed)
  for i in xrange(num_labels):
    a = images[labels == i]
    if n_per_label == -1:  # use all available labeled data
      res.append(a)
    else:  # use randomly chosen subset
      res.append(a[rng.choice(len(a), n_per_label, False)])
  return res


def create_virt_emb(n, size):
  """Create virtual embeddings."""
  emb = slim.variables.model_variable(
      name='virt_emb',
      shape=[n, size],
      dtype=tf.float32,
      trainable=True,
      initializer=tf.random_normal_initializer(stddev=0.01))
  return emb


def confusion_matrix(labels, predictions, num_labels):
  """Compute the confusion matrix."""
  rows = []
  for i in xrange(num_labels):
    row = np.bincount(predictions[labels == i], minlength=num_labels)
    rows.append(row)
  return np.vstack(rows)


class SemisupModel(object):
  """Helper class for setting up semi-supervised training."""

  def __init__(self, model_func, num_labels, input_shape, test_in=None):
    """Initialize SemisupModel class.

    Creates an evaluation graph for the provided model_func.

    Args:
      model_func: Model function. It should receive a tensor of images as
          the first argument, along with the 'is_training' flag.
      num_labels: Number of taget classes.
      input_shape: List, containing input images shape in form
          [height, width, channel_num].
      test_in: None or a tensor holding test images. If None, a placeholder will
        be created.
    """

    self.num_labels = num_labels
    #self.step = slim.get_or_create_global_step()
    self.step = [tf.Variable(0, trainable=False), tf.Variable(0, trainable=False)]
    #self.ema = tf.train.ExponentialMovingAverage(0.99, self.step[0])

    self.test_batch_size = 100

    self.model_func = model_func

    if test_in is not None:
      self.test_in = test_in
    else:
      self.test_in = tf.placeholder(np.float32, [self.test_batch_size] + input_shape, 'test_in')


    self.test_emb, self.test_shared, self.test_logit = [], [], []
    mkb_reuse = False
    for i in range(len(self.model_func)):
        test_emb_0, _ = self.image_to_embedding(i, self.test_in, is_training=False, is_mkb_reuse=mkb_reuse)
        test_logit_0 = self.embedding_to_logit(i, test_emb_0, is_training=False)
        self.test_emb.append(test_emb_0)
        #self.test_shared.append(test_shared_0)
        self.test_logit.append(test_logit_0)
        mkb_reuse = True

    self.logit_loss, self.kl_loss, self.mse_loss, self.cosine_loss = [], [], [], []
    for i in range(len(self.model_func)):
        self.logit_loss.append([])
        #self.mse_loss.append(None)
        self.cosine_loss.append(None)
        self.kl_loss.append([])

  def image_to_embedding(self, net_id, images, is_training=True, is_mkb_reuse=True):
    """Create a graph, transforming images into embedding vectors."""
   #with tf.variable_scope('net_'+str(net_id), reuse=is_training):
    return self.model_func[net_id](images, net_id, is_training=is_training, is_mkb_reuse=is_mkb_reuse)


  def embedding_to_logit(self, net_id, embedding, is_training=True):
    """Create a graph, transforming embedding vectors to logit classs scores."""
    with tf.variable_scope('net_' + str(net_id), reuse=is_training):
      return slim.fully_connected(
          embedding,
          self.num_labels,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(1e-4))
          #weights_regularizer=None)

  def shared_to_embedding(self, net_id, embedding, emb_size=128, is_training=True):
    """Create a graph, transforming embedding vectors to logit classs scores."""
    with tf.variable_scope('net_'+str(net_id)+'/fc_shared', reuse=is_training):
       emb =  slim.fully_connected(
          embedding,
          emb_size,
          activation_fn=tf.nn.elu,
          weights_regularizer=slim.l2_regularizer(1e-3))
          #weights_regularizer=None)
       emb =  slim.fully_connected(
          emb,
          self.num_labels,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(1e-4))
       return emb

  def shared_embedding_to_logit(self, net_id, embedding, is_training=True):
    """Create a graph, transforming embedding vectors to logit classs scores."""
    with tf.variable_scope('net_' + str(net_id) + '/shared_logit', reuse=is_training):
      return slim.fully_connected(
          embedding,
          self.num_labels,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(1e-4))
          #weights_regularizer=None)

  def embedding_to_shared_256(self, embedding, is_training=True):
    """Map embeddings using the same fc layer"""
    with tf.variable_scope('shared_layer', reuse=is_training):
      #emb_size = embedding.get_shape().as_list()[1]
      emb_size = [256, 128]
      print('emb_size', emb_size)
      s1 =  slim.fully_connected(
          embedding,
          emb_size[0],
          activation_fn=tf.nn.elu,
          weights_regularizer=slim.l2_regularizer(1e-3))
          #weights_regularizer=None)

      s2 =  slim.fully_connected(
          s1,
          emb_size[1],
          activation_fn=tf.nn.elu,
          weights_regularizer=slim.l2_regularizer(1e-3))
          #weights_regularizer=None)
      return s2

  def embedding_to_shared(self, embedding, is_training=True):
    """Map embeddings using the same fc layer"""
    with tf.variable_scope('shared_layer', reuse=is_training):
      #emb_size = embedding.get_shape().as_list()[1]
      emb_size = 128
      print('emb_size', emb_size)
      return slim.fully_connected(
          embedding,
          emb_size,
          activation_fn=tf.nn.elu,
          weights_regularizer=slim.l2_regularizer(1e-3))
          #weights_regularizer=None)

  def add_semisup_loss(self, a, b, labels, net_id, walker_weight=1.0, visit_weight=1.0):
    """Add semi-supervised classification loss to the model.

    The loss constist of two terms: "walker" and "visit".

    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      labels : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """

    with tf.name_scope('walk_loss_' + str(net_id)):
        equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
        equality_matrix = tf.cast(equality_matrix, tf.float32)
        p_target = (equality_matrix / tf.reduce_sum(
            equality_matrix, [1], keep_dims=True))

        match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
        p_ab = tf.nn.softmax(match_ab, name='p_ab')
        p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
        p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

        self.create_walk_statistics(p_aba, equality_matrix)

        loss_aba = tf.losses.softmax_cross_entropy(
            p_target,
            tf.log(1e-8 + p_aba),
            weights=walker_weight,
            scope='walk_loss_' + str(net_id))

        print(loss_aba)
        self.walk_loss[net_id] = loss_aba

        self.add_visit_loss(p_ab, net_id, visit_weight)

        tf.summary.scalar('Walk_Loss_' + str(net_id), loss_aba)

  def add_semisup_loss_single(self, a, b, labels_a, labels_b, net_id, walker_weight=1.0, visit_weight=1.0):
    """Add semi-supervised classification loss to the model.

    The loss constist of two terms: "walker" and "visit".

    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      labels_a : [N] tensor with labels for supervised embeddings.
      labels_b : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """

    with tf.name_scope('walk_loss_' + str(net_id)):
        equality_matrix = tf.equal(tf.reshape(labels_a, [-1, 1]), labels_b)
        equality_matrix = tf.cast(equality_matrix, tf.float32)
        p_target = (equality_matrix / tf.reduce_sum(
            equality_matrix, [1], keep_dims=True))

        match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
        p_ab = tf.nn.softmax(match_ab, name='p_ab')

        #self.create_walk_statistics(p_aba, equality_matrix)

        loss_ab = tf.losses.softmax_cross_entropy(
            p_target,
            tf.log(1e-8 + p_ab),
            weights=walker_weight,
            scope='walk_loss_' + str(net_id))

        self.walk_loss[net_id] = loss_ab

        #self.add_visit_loss(p_ab, net_id, visit_weight)

        tf.summary.scalar('Walk_Loss_' + str(net_id), loss_ab)

  def add_visit_loss(self, p, net_id, weight=1.0):
    """Add the "visit" loss to the model.

    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      weight: Loss weight.
    """
    visit_probability = tf.reduce_mean(
        p, [0], keep_dims=True, name='visit_prob')
    t_nb = tf.shape(p)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='visit_loss_' + str(net_id))

    self.visit_loss[net_id] = visit_loss

    tf.summary.scalar('Visit_Loss_' + str(net_id), visit_loss)

  def add_logit_loss(self, net_id, logits, labels, weight=1.0, smoothing=0.0):
    """Add supervised classification loss to the model."""
    logit_loss = tf.losses.softmax_cross_entropy(
        tf.one_hot(labels, logits.get_shape()[-1]),
        logits,
        scope='loss_logit_'+str(net_id) + '_' + str(len(self.logit_loss[net_id])),
        weights=weight,
        label_smoothing=smoothing)
    self.logit_loss[net_id].append(logit_loss)

    tf.summary.scalar('Loss_Logit_Net_'+str(net_id)+'_'+str(len(self.logit_loss[net_id])), logit_loss)

  def add_kl_loss(self, net_id, logits_1, logits_2, weight=1.0, smoothing=0.0):
    """Add KL divergence loss to the model."""
    constant = 0.00001
    prob_1 = tf.nn.softmax(logits_1)  + constant
    prob_2 = tf.nn.softmax(logits_2)  + constant
    kl_loss = weight * tf.reduce_sum(prob_1 * tf.log(prob_1/prob_2))
    self.kl_loss[net_id].append(kl_loss)

    tf.summary.scalar('Loss_KL_Net_'+str(net_id) + '_' + str(len(self.kl_loss[net_id])), kl_loss)

  def add_mse_loss(self, logits_1, logits_2, l_num, weight=1.0, smoothing=0.0):
    """Add mse divergence loss to the model."""
    if l_num == 'l1':
        #mse_loss = weight * tf.reduce_sum(tf.abs(logits_1-logits_2)**l_num)
        dist = tf.reduce_sum(tf.abs(tf.subtract(logits_1, logits_2)), 1)
        mse_loss = weight * tf.reduce_mean(tf.maximum(dist, 0.0), 0)
    elif l_num == 'l2':
        """
        dist = tf.reduce_sum(tf.square(tf.subtract(logits_1, logits_2)), 1)
        mse_loss = weight * tf.reduce_mean(tf.maximum(dist, 0.0), 0)
        """
        mse_loss = weight * tf.reduce_mean(tf.square(logits_1-logits_2))

    self.mse_loss.append(mse_loss)
    tf.summary.scalar('Loss_MSE_MKB_'+str(len(self.mse_loss)), mse_loss)

  def add_cosine_loss(self, net_id, logits_1, logits_2, weight=1.0, smoothing=0.0):
    """Add supervised classification loss to the model."""
    cosine_loss = tf.losses.cosine_distance(
        logits_1,
        logits_2,
        dim=1,
        scope='cosine_'+str(net_id),
        weights=weight)
    self.cosine_loss[net_id] = cosine_loss

    tf.summary.scalar('Loss_Cosine_Net_'+str(net_id), cosine_loss)

  def create_walk_statistics(self, p_aba, equality_matrix):
    """Adds "walker" loss statistics to the graph.

    Args:
      p_aba: [N, N] matrix, where element [i, j] corresponds to the
          probalility of the round-trip between supervised samples i and j.
          Sum of each row of 'p_aba' must be equal to one.
      equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
          i and j belong to the same class.
    """
    # Using the square root of the correct round trip probalilty as an estimate
    # of the current classifier accuracy.
    per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1)**0.5
    estimate_error = tf.reduce_mean(
        1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')
    self.add_average(estimate_error)
    self.add_average(p_aba)

    tf.summary.scalar('Stats_EstError', estimate_error)

  def add_average(self, variable):
    """Add moving average variable to the model."""
    #print('-------------------tf.GraphKeys.UPDATE_OPS')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.ema.apply([variable]))
    #print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    average_variable = tf.identity(
        self.ema.average(variable), name=variable.name[:-2] + '_avg')
    return average_variable

  def create_train_op(self, learning_rate):
    """Create and return training operation."""

    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    print('=============losses=======================')
    print(tf.losses.get_losses())

    self.train_loss = tf.losses.get_total_loss()
    #self.train_loss_average = self.add_average(self.train_loss)

    tf.summary.scalar('Learning_Rate', learning_rate)
    #tf.summary.scalar('Loss_Total_Avg', self.train_loss_average)
    tf.summary.scalar('Loss_Total', self.train_loss)

    #trainer = tf.train.AdamOptimizer(learning_rate)
    trainer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    self.train_op = slim.learning.create_train_op(self.train_loss, trainer, global_step=self.step[0])
    return self.train_op

  def create_train_op_type(self, learning_rate, tp='kl'):
    """Create and return training operation."""

    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    print('=============losses=======================')
    print(tf.losses.get_losses())

    if tp == 'kl':
        self.train_loss = tf.losses.get_total_loss() + tf.add_n(self.kl_loss[0] + self.kl_loss[1])
    elif tp == 'mse':
        self.train_loss = tf.losses.get_total_loss() + self.mse_loss[0]
    elif tp == 'all':
        self.train_loss = tf.losses.get_total_loss() + tf.add_n(self.mse_loss + self.kl_loss[0] + self.kl_loss[1])
    elif tp == 'cosine':
        self.train_loss = tf.losses.get_total_loss()


    #self.train_loss_average = self.add_average(self.train_loss)

    tf.summary.scalar('Learning_Rate', learning_rate)
    #tf.summary.scalar('Loss_Total_Avg', self.train_loss_average)
    tf.summary.scalar('Loss_Total', self.train_loss)

    #trainer = tf.train.AdamOptimizer(learning_rate)
    trainer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    self.train_op = slim.learning.create_train_op(self.train_loss, trainer, global_step=self.step[0])
    return self.train_op


  def create_separate_train_op(self, learning_rates, tp='kl'):
    """Create and return training operation."""

    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

    print('=============losses=======================')
    print(tf.losses.get_losses())
    print(tf.losses.get_regularization_losses())
    all_regularization_losses = tf.losses.get_regularization_losses()

    regular_losses_0 = [l for l in all_regularization_losses if ('net_0' in l.name or 'clm_shared' in l.name)]
    regular_losses_1 = [l for l in all_regularization_losses if ('net_1' in l.name or 'clm_shared' in l.name)]
    #regular_losses_0 = [l for l in all_regularization_losses if ('net_0' in l.name)]
    #regular_losses_1 = [l for l in all_regularization_losses if ('net_1' in l.name)]

    print(regular_losses_0)
    print(regular_losses_1)


    self.train_loss = [None, None]
    if tp == 'kl':
        self.train_loss[0] = tf.add_n(self.logit_loss[0] + self.kl_loss[0] + [self.cosine_loss[0]] + regular_losses_0)
        self.train_loss[1] = tf.add_n(self.logit_loss[1] + self.kl_loss[1] + [self.cosine_loss[1]] + regular_losses_1)
    elif tp == 'mse':
        self.train_loss[0] = tf.add_n(self.logit_loss[0] + [self.mse_loss[0]]  + regular_losses_0)
        self.train_loss[1] = tf.add_n(self.logit_loss[1] + [self.mse_loss[1]]  + regular_losses_1)
    elif tp == 'all':
        self.train_loss[0] = tf.add_n(self.logit_loss[0] + self.kl_loss[0] + [self.mse_loss[0]]  + regular_losses_0)
        self.train_loss[1] = tf.add_n(self.logit_loss[1] + self.kl_loss[1] + [self.mse_loss[1]]  + regular_losses_1)
    elif tp == 'cosine':
        pass
        #self.train_loss = tf.losses.get_total_loss()


    tf.summary.scalar('Learning_Rate_Net_0', learning_rates[0])
    tf.summary.scalar('Learning_Rate_Net_1', learning_rates[1])
    tf.summary.scalar('Loss_Total_Net_0', self.train_loss[0])
    tf.summary.scalar('Loss_Total_Net_1', self.train_loss[1])

    self.trainer = [None, None]
    self.trainer[0] = tf.train.MomentumOptimizer(learning_rate=learning_rates[0], momentum=0.9)
    self.trainer[1] = tf.train.MomentumOptimizer(learning_rate=learning_rates[1], momentum=0.9)

    self.train_op = [None, None]
    print('--------------trained variables for net 0--------------------------')
    trained_vars_0 = [v for v in tf.trainable_variables() if ('net_0' in v.name or 'clm_shared' in v.name)]
    #trained_vars_0 = [v for v in tf.trainable_variables() if ('net_0' in v.name)]
    print(trained_vars_0)
    self.train_op[0] = slim.learning.create_train_op(self.train_loss[0], self.trainer[0], global_step=self.step[0], variables_to_train=trained_vars_0)

    print('--------------trained variables for net 1--------------------------')
    trained_vars_1 = [v for v in tf.trainable_variables() if ('net_1' in v.name or 'clm_shared' in v.name)]
    #trained_vars_1 = [v for v in tf.trainable_variables() if ('net_1' in v.name)]
    print(trained_vars_1)
    self.train_op[1] = slim.learning.create_train_op(self.train_loss[1], self.trainer[1], global_step=self.step[1], variables_to_train=trained_vars_1)
    return self.train_op


  def calc_embedding(self, images, endpoint):
    """Evaluate 'endpoint' tensor for all 'images' using batches."""
    batch_size = self.test_batch_size
    emb = []
    for i in xrange(0, len(images), batch_size):
      emb.append(endpoint.eval({self.test_in: images[i:i + batch_size]}))
    return np.concatenate(emb)

  def classify(self, images):
    """Compute logit scores for provided images."""
    return (self.calc_embedding(images, self.test_logit[0]), self.calc_embedding(images, self.test_logit[1]))

  def classify_single(self, images):
    """Compute logit scores for provided images."""
    return self.calc_embedding(images, self.test_logit[0])

  def load_vgg_weights(self, net_id, sess):
    weights = np.load("/home/amax/cxt/pr/face_svm_tf/VGGNet/vgg16_weights.npz")
    keys = sorted(weights.keys())
    self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net_"+str(net_id))
    print(self.params)
    for i, k in enumerate(keys):
        if i > 25: break
        print(i, k, weights[k].shape)
        sess.run(self.params[i].assign(weights[k]))

  def save_shared_weights(self, save_path, sess):
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shared_layer")
    print('------------saveing shared weights -------------------------')
    weights = []
    for param in params:
        print(param)
        weights.append(sess.run(param))
    np.save(save_path, weights)

  def load_shared_weights(self, load_path, sess):
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="shared_layer")
    print('------------loading shared weights -------------------------')
    weights = np.load(load_path)
    for i in range(len(params)):
        print(params[i])
        print(weights[i])
        sess.run(params[i].assign(weights[i]))

