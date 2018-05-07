import numpy as np
import cPickle

NUM_LABELS = 10
NUM_LABELS_100 = 100
IMAGE_SHAPE = [32, 32, 3]

def get_dataset_cifar100(subset):
    if subset == 'training':
        dataset_file = '/home/amax/cxt/clm/data/cifar-100-python/train'
    else:
        dataset_file = '/home/amax/cxt/clm/data/cifar-100-python/test'
    with open(dataset_file, 'rb') as fo:
        dataset = cPickle.load(fo)

    # construct one-hot vector for labels
    images = dataset['data']
    images = np.reshape(images, (len(images), 3, 32, 32))
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(dataset['fine_labels'])

    if subset == 'training':
        train_images, train_labels = images[:45000], labels[:45000]
        valid_images, valid_labels = images[45000:], labels[45000:]
        pad_width = ((0, 0), (2, 2), (2, 2), (0, 0))
        train_images = np.pad(train_images, pad_width=pad_width, mode='constant', constant_values=0)
        return train_images, train_labels, valid_images, valid_labels
    else:
        return images, labels

    #return images, labels,  dataset['coarse_labels'], dataset['filenames']

def get_dataset_cifar10(subset):
    if subset == 'training':
        print('Fetching the training dataset ...')
        for i in range(5):
            f = open('/home/amax/cxt/cll/data/cifar-10-batches-py/data_batch_' + str(i+1), 'rb')
            dic = cPickle.load(f)
            if i == 0:
                images, labels = dic['data'], dic['labels']
            else:
                images = np.concatenate((images, dic['data']), axis=0)
                labels += dic['labels']
    else:
        print('Fetching the test dataset ...')
        f = open('/home/amax/cxt/cll/data/cifar-10-batches-py/test_batch', 'rb')
        dic = cPickle.load(f)
        images, labels = dic['data'], dic['labels']

    images = np.reshape(images, (len(images), 3, 32, 32))
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(labels)

    if subset == 'training':
        train_images, train_labels = images[:45000], labels[:45000]
        valid_images, valid_labels = images[45000:], labels[45000:]
        pad_width = ((0, 0), (2, 2), (2, 2), (0, 0))
        train_images = np.pad(train_images, pad_width=pad_width, mode='constant', constant_values=0)
        return train_images, train_labels, valid_images, valid_labels
    else:
        return images, labels
