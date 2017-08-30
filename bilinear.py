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

	# idx__ : [ batch_size, channels, _, 2 ], 2 = (#batch, #channel)
	mesh = tf.meshgrid(tf.range(x_shape[1]), tf.range(x_shape[0]))
	idx = tf.expand_dims(tf.stack([mesh[1], mesh[0]],-1), 2)
	idx_h = tf.tile(idx, [1, 1, x_shape[2], 1])
	idx_w = tf.tile(idx, [1, 1, x_shape[3], 1])

	coords_0_ = tf.concat([idx_h, tf.expand_dims(tf.cast(tf.floor(coords_h), tf.int32), -1)], -1)
	coords_1_ = tf.concat([idx_h, tf.expand_dims(tf.cast(tf.ceil(coords_h), tf.int32), -1)], -1)
	coords__0 = tf.concat([idx_w, tf.expand_dims(tf.cast(tf.floor(coords_w), tf.int32), -1)], -1)
	coords__1 = tf.concat([idx_w, tf.expand_dims(tf.cast(tf.ceil(coords_w), tf.int32), -1)], -1)

	vals_0_ = tf.matrix_transpose(tf.gather_nd(x, coords_0_))
	vals_1_ = tf.matrix_transpose(tf.gather_nd(x, coords_1_))

	vals_00 = tf.gather_nd(vals_0_, coords__0)
	vals_01 = tf.gather_nd(vals_0_, coords__1)
	vals_10 = tf.gather_nd(vals_1_, coords__0)
	vals_11 = tf.gather_nd(vals_1_, coords__1)

	# coords_w = tf.tile(tf.expand_dims(coords_w, 2), [1, 1, x_shape[2], 1])
	# coords_h = tf.tile(tf.expand_dims(coords_h, 3), [1, 1, 1, x_shape[3]])
	coords_x = tf.expand_dims(coords_w - tf.floor(coords_w), 3)
	coords_y = tf.expand_dims(coords_h - tf.floor(coords_h), 2)

	vals = vals_00 + \
				 (vals_10 - vals_00) * coords_x + \
				 (vals_01 - vals_00) * coords_y + \
				 (vals_11 + vals_00 - vals_10 - vals_01) * coords_x * coords_y

	return vals


# original feature map
grid = tf.random_normal(shape=(batch_size, channels, feature_height, feature_width), seed=11111)

weights_w = tf.reduce_sum(cl.conv2d(grid, num_outputs=channels, kernel_size=[1, 7], stride=1, 
											activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW'), axis=2)
weights_h = tf.reduce_sum(cl.conv2d(grid, num_outputs=channels, kernel_size=[7, 1], stride=1, 
											activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW'), axis=3)

coords_w = weights_to_coords(weights_w)
coords_h = weights_to_coords(weights_h)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
kkk = sess.run(vals)
print (kkk.shape)

