###################################################################################################
#
# ECE 532 Final Project Fall 2020
# Timothy Lieb tlieb@wisc.edu
#

import numpy

# Method for reading in the "pickled" object images
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Read in the datasets 5 training batches and 1 test batch, each has 10,000 images
data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
test_batch = unpickle('cifar-10-batches-py/test_batch')

# Each data_batch is a dictionary with the following items
# b'batch_label --> specifies which batch it is
# b'labels --> array of 10,000 labels 0-9 correspoding to the correct classification
# b'data --> 10,000 x 3072 array of uint8 pixels, each rows is a 32x32 image with the first 1024 entries being the red,
#            the second 1024 entries being the green, and the last 1024 entries being the blue

db1_labels = data_batch_1[b'labels']
db1_data = data_batch_1[b'data']
db2_labels = data_batch_2[b'labels']
db2_data = data_batch_2[b'data']
db3_labels = data_batch_3[b'labels']
db3_data = data_batch_3[b'data']
db4_labels = data_batch_4[b'labels']
db4_data = data_batch_4[b'data']
db5_labels = data_batch_5[b'labels']
db5_data = data_batch_5[b'data']
tb_labels = test_batch[b'labels']
tb_data = test_batch[b'data']

# print(db1_data[0,:]) # --> first image
# print(db1_data[:,0]) # -->first column


###################################################################################################
#
# Ridge Regression
#
def ridge_regression():




###################################################################################################
#
# K-Mean Clustering? --> Is this going to work?
#




###################################################################################################
#
# Neural Networks
#