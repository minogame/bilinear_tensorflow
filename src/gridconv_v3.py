import tensorflow as tf
import tensorflow.contrib.layers as cl

def weights_to_coords(x):
	# both inputs and outputs are [ batch_size, channels, size ]
	size = x.get_shape().as_list()[2]
	norm = tf.reduce_sum(x, axis=2, keep_dims=True)
	x = size * x / norm
	y = x

	x = tf.expand_dims(x, -1)
	x = tf.tile(x, multiples=[1, 1, 1, size])
	x = tf.matrix_band_part(x, 0, -1) # upper triangular part of x
	x = tf.reduce_sum(x, axis=2) - y
	x = tf.stop_gradient(x) + y/2.0
	return x

def batch_bilinear(x, weights_w, weights_h):
	# x: [ batch_size, channels, height, width ]
	x_shape = x.get_shape().as_list()

	# coords_w: [ batch_size, channels, width ]
	# coords_h: [ batch_size, channels, height ]
	coords_w = weights_to_coords(weights_w)
	coords_h = weights_to_coords(weights_h)
	tf.add_to_collection('70f92c137c01d89c6477c5ef22411bfe', [coords_w, coords_h])

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

	coords_x = tf.expand_dims(coords_w - tf.floor(coords_w), 3)
	coords_y = tf.expand_dims(coords_h - tf.floor(coords_h), 2)

	vals = vals_00 + \
				 (vals_10 - vals_00) * coords_x + \
				 (vals_01 - vals_00) * coords_y + \
				 (vals_11 + vals_00 - vals_10 - vals_01) * coords_x * coords_y

	return vals

def gridconv2d( x, scope,
								num_outputs, kernel_size, stride=1, length=5,
								activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
								normalizer_fn=None, normalizer_params=None, one_c=False,
								weights_initializer=cl.variance_scaling_initializer()):
	# The network is built based on 'NCHW'.
	x_shape = x.get_shape().as_list()
	c = 1 if one_c else x_shape[1]

	with tf.variable_scope(scope):
		# weights_w = tf.reduce_sum(cl.conv2d(x, num_outputs=c, kernel_size=[1, length], stride=1, 
		# 											activation_fn=tf.nn.sigmoid, padding='SAME', data_format='NCHW', 
		# 											#normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, 
		# 											scope='_W'), axis=2)
		# weights_h = tf.reduce_sum(cl.conv2d(x, num_outputs=c, kernel_size=[length, 1], stride=1,
		# 											activation_fn=tf.nn.sigmoid, padding='SAME', data_format='NCHW', 
		# 											#normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
		# 											scope='_H'), axis=3)

		weights = cl.conv2d(x, num_outputs=c, kernel_size=[3, 3], stride=1,
													activation_fn=tf.nn.sigmoid, padding='SAME', data_format='NCHW', 
													normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
													scope='GRID')
		# weights_w = tf.reciprocal(tf.reduce_sum(weights, axis=2))
		# weights_h = tf.reciprocal(tf.reduce_sum(weights, axis=3))
		weights_w = tf.reduce_sum(weights, axis=2)
		weights_h = tf.reduce_sum(weights, axis=3)

		if one_c:
			weights_w = tf.tile(weights_w, [1, x_shape[1], 1])
			weights_h = tf.tile(weights_h, [1, x_shape[1], 1])

		# weights_w = tf.Print(weights_w, [weights_w], summarize=32)
		# tf.add_to_collection('70f92c137c01d89c6477c5ef22411bfe', [weights_w, weights_h])

		x = batch_bilinear(x, weights_w, weights_h)
		x = cl.conv2d(x, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride,
									activation_fn=activation_fn, padding=padding,
									normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
									weights_initializer=weights_initializer, data_format='NCHW', scope='MAIN')

		return x