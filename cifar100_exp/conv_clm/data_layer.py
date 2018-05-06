import os
import re
import sys

import tensorflow as tf
import tensorlayer as tl
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle

from scipy.misc import imread, imresize, imsave
from scipy.misc.pilutil import imshow

import utils
from tools import mnist as mnist_tools
from tools import stl10 as stl10_tools
from tools import cifar as cifar_tools

import backend



class data_layer:
    def __init__(self, dataset):
        # Get cifar data
        print('-----------------------------data_layer--------------------------------')
        self.dataset = dataset
        if dataset == 'cifar10':
            print('--------loading cifar10 dataset------------------')
            self.train_images, self.train_labels, self.valid_images, self.valid_labels  = cifar_tools.get_dataset_cifar10('training')
            print(self.train_images.shape, self.train_labels.shape)
            print(self.valid_images.shape, self.valid_labels.shape)
            self.test_images, self.test_labels = cifar_tools.get_dataset_cifar10('test')
            print(self.test_images.shape, self.test_labels.shape)
            self.sup_by_label = backend.sample_by_label(self.train_images, self.train_labels, -1, 10, None)
        elif dataset == 'cifar100':
            print('--------loading cifar100 dataset------------------')
            self.train_images, self.train_labels, self.valid_images, self.valid_labels  = cifar_tools.get_dataset_cifar100('training')
            print(self.train_images.shape, self.train_labels.shape)
            print(self.valid_images.shape, self.valid_labels.shape)
            self.test_images, self.test_labels = cifar_tools.get_dataset_cifar100('test')
            print(self.test_images.shape, self.test_labels.shape)
            self.sup_by_label = None
        elif dataset == 'mnist':
            print('--------loading mnist dataset------------------')
            self.train_images, self.train_labels = mnist_tools.get_data('train')
            print(self.train_images.shape, self.train_labels.shape)
            self.test_images, self.test_labels = mnist_tools.get_data('test')
            print(self.test_images.shape, self.test_labels.shape)
            self.sup_by_label = None
        elif dataset == 'stl10':
            print('--------loading stl10 dataset------------------')
            self.trainval_images, self.trainval_labels = stl10_tools.get_data('training')
            self.train_images, self.train_labels = self.trainval_images[:4500], self.trainval_labels[:4500]
            self.valid_images, self.valid_labels = self.trainval_images[4500:], self.trainval_labels[4500:]
            print(self.train_images.shape, self.train_labels.shape)
            print(self.valid_images.shape, self.valid_labels.shape)
            self.test_images, self.test_labels = stl10_tools.get_data('test')
            print(self.test_images.shape, self.test_labels.shape)
            self.sup_by_label = None


        self.it = [0, 0]
        self.perm = [[], []]
        self.batch_size = 64
        print('-----------------------------data_layer--------------------------------')

    def get_next_batch(self, net_id):

        if self.it[net_id] == 0 or self.it[net_id] + self.batch_size - 1 >= len(self.train_images):
            self.perm[net_id] = np.random.permutation(len(self.train_images))
            self.it[net_id] = 0

        #print('get minibatch for net ' + str(net_id) + ": , iter: " + str(self.it[net_id]))

        batch_index = self.perm[net_id][self.it[net_id]:self.it[net_id]+self.batch_size]

        batch_ims = self.train_images[batch_index]
        batch_labels = self.train_labels[batch_index]

        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            batch_ims = utils.random_crop_and_flip(batch_ims, 2)
            #batch_ims = utils.whitening_image(batch_ims)
        if self.dataset == 'stl10':
            pad_width = ((0, 0), (4, 4), (4, 4), (0, 0))
            batch_ims = np.pad(batch_ims, pad_width=pad_width, mode="constant", constant_values=0)
            batch_ims = utils.random_crop_and_flip(batch_ims, 4)

        self.it[net_id] += self.batch_size
        return batch_ims, batch_labels

    def get_batch_per_class(self, num_per_class):
        class_labels = np.arange(len(self.sup_by_label))
        batch_images, batch_labels = [], []
        for images, label in zip(self.sup_by_label, class_labels):
            labels = np.full((num_per_class), label)
            images = images[np.random.choice(len(images), num_per_class)]
            batch_images.append(images)
            batch_labels.append(labels)
        return np.concatenate(batch_images, axis=0), np.concatenate(batch_labels, axis=0)

    def get_train_set(self):
        return self.train_images, self.train_labels

    def get_valid_set(self):
        return self.valid_images, self.valid_labels

    def get_test_set(self):
        return self.test_images, self.test_labels


if __name__ == '__main__':
    dl = data_layer('stl10')
    for i in range(10):
        image, labels = dl.get_batch_per_class(10)
        print(type(image))
        print(image.shape)
        print(image.sum())
        print(labels)
        raw_input()






