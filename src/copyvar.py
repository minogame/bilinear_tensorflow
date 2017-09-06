import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
import re

import tensorflow as tf
import tensorflow.contrib.layers as cl
from model import trash_cnn_cifar, normal_cnn_cifar

x = tf.random_normal(shape=(1,3,32,32))

normal_cnn = normal_cnn_cifar(name='aaaa')
grid_cnn = trash_cnn_cifar(name='grid')

n = normal_cnn(x, is_training=tf.constant(True))
g = grid_cnn(x, is_training=tf.constant(True))

normal_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='aaaa')
grid_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='grid')
grid_main_list = [v for v in grid_list if not 'GRID' in v.name]

n_saver = tf.train.Saver(var_list=normal_list)
g_saver = tf.train.Saver(var_list=grid_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_saver.restore(sess, tf.train.latest_checkpoint('ckpt/normal_cnn'))

for nv, gv in zip(normal_list, grid_main_list):
	op = tf.assign(gv, nv)

	sess.run(op)

	print ('{} -> {}'.format(nv.name, gv.name))

g_saver.save(sess, 'ckpt/grid_cnn/epoch', global_step=0)

# print (normal_list)

# print (grid_main_list)