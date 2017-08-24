import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
import tensorflow.contrib.layers as cl

# def bilinear(coords, grid):
# 	'''
# 	coords: tensor, shape = [ coord_x, coord_y ]
# 	grid: tensor, shape = [ size_x, size_y ]

# 	return: tensor, shape = [ idx, value ]
# 	'''

batch_size = 256
feature_height = 72
feature_width = 56
channels = 96

# original feature map
grid = tf.random_normal(shape=(batch_size, feature_height, feature_width, channels))

def lrelu(x, leak=0.2, name='lrelu'):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

weights_x = cl.conv2d(grid, num_outputs=channels, kernel_size=[1, 7], stride=1, activation_fn=lrelu, padding='SAME')
weights_y = cl.conv2d(grid, num_outputs=channels, kernel_size=[1, 7], stride=1, activation_fn=lrelu, padding='SAME')