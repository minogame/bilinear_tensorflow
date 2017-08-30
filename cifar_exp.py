import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)

import tensorflow as tf

# Data loading and preprocessing
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.datasets import cifar10
(cifar10_X, cifar10_Y), (X_test, Y_test) = cifar10.load_data()
cifar10_X, cifar10_Y = shuffle(cifar10_X, cifar10_Y)
cifar10_Y = to_categorical(cifar10_Y, 10)
Y_test = to_categorical(Y_test, 10)