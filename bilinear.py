import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
import tensorflow.contrib.layers as cl

# The network is built based on 'NCHW'.

batch_size = 256
channels = 96
feature_height = 72
feature_width = 56

# original feature map
grid = tf.random_normal(shape=(batch_size, channels, feature_height, feature_width), seed=11111)

def lrelu(x, leak=0.2, name='lrelu'):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def weights_to_coords(x):
	# both inputs and outputs are [ batch_size, channels, size ]
	size = x.get_shape().as_list()[2]
	norm = tf.reduce_sum(x, axis=2, keep_dims=True)
	x = size * x / norm
	y = x

	x = tf.expand_dims(x, -1)
	x = tf.tile(x, multiples=[1, 1, 1, size])
	x = tf.matrix_band_part(x, 0, -1) # upper triangular part of x
	x = tf.reduce_sum(x, axis=2) - y/2.0
	return x

def batch_bilinear(x, coords_w, coords_h):
	# x: [ batch_size, channels, height, width ]
	# coords_w: [ batch_size, channels, width ]
	# coords_h: [ batch_size, channels, height ]
	x_shape = x.get_shape().as_list()

	coords_w = tf.tile(tf.expand_dims(coords_w, 2), [1, 1, x_shape[2], 1])
	coords_h = tf.tile(tf.expand_dims(coords_h, 3), [1, 1, 1, x_shape[3]])

	coords__0 = tf.cast(tf.floor(coords_w), tf.int32)
	coords__1 = tf.cast(tf.ceil(coords_w), tf.int32)
	coords_0_ = tf.cast(tf.floor(coords_h), tf.int32)
	coords_1_ = tf.cast(tf.ceil(coords_h), tf.int32)

	f_00 = 

# 	coords_lt = tf.cast(tf.floor(coords), 'int32')
# 	coords_rb = tf.cast(tf.ceil(coords), 'int32')
# 	coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
# 	coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

# 	vals_lt = tf.gather_nd(input, coords_lt)
# 	vals_rb = tf.gather_nd(input, coords_rb)
# 	vals_lb = tf.gather_nd(input, coords_lb)
# 	vals_rt = tf.gather_nd(input, coords_rt)

	return
	


weights_w = tf.reduce_sum(cl.conv2d(grid, num_outputs=channels, kernel_size=[1, 7], stride=1, 
											activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW'), axis=2)
weights_h = tf.reduce_sum(cl.conv2d(grid, num_outputs=channels, kernel_size=[7, 1], stride=1, 
											activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW'), axis=3)

coords_w = weights_to_coords(weights_w)
coords_h = weights_to_coords(weights_h)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# print (sess.run(coords_x)[0][0])

# def tf_map_coordinates(input, coords, order=1):
# 	"""Tensorflow verion of scipy.ndimage.map_coordinates

# 	Note that coords is transposed and only 2D is supported

# 	Parameters
# 	----------
# 	input : tf.Tensor. shape = (s, s)
# 	coords : tf.Tensor. shape = (n_points, 2)
# 	"""

# 	assert order == 1

# 	coords_lt = tf.cast(tf.floor(coords), 'int32')
# 	coords_rb = tf.cast(tf.ceil(coords), 'int32')
# 	coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
# 	coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

# 	vals_lt = tf.gather_nd(input, coords_lt)
# 	vals_rb = tf.gather_nd(input, coords_rb)
# 	vals_lb = tf.gather_nd(input, coords_lb)
# 	vals_rt = tf.gather_nd(input, coords_rt)

# 	coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
# 	vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
# 	vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
# 	mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

# 	return mapped_vals

# test_input = grid[0][0]
# grad_input = tf.random_uniform(shape=[100], seed=123)
# test_coords = tf.random_uniform(shape=[100, 2], seed=1234)

# a = tf_map_coordinates(test_input, test_coords)
# b = tf.gradients(a, test_coords, grad_ys=grad_input)

# print (sess.run(b))