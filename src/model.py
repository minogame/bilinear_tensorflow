import tensorflow as tf
import tensorflow.contrib.layers as cl

from gridconv import gridconv2d

# # # # # # # # # CIFAR # # # # # # # # #

def normal_cnn_cifar(x, name, is_training, reuse=False):
	# The network is built based on 'NCHW'.
	bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}

	with tf.variable_scope(name, reuse=reuse):
		x = cl.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv1_1')
		x = cl.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv1_2')
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = cl.conv2d(x, num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv2_1')
		x = cl.conv2d(x, num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv2_2')
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = cl.conv2d(x, num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv3_1')
		x = cl.conv2d(x, num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv3_2')
		x = tf.reduce_mean(x, [2, 3])
		x = cl.fully_connected(x, num_outputs=10, activation_fn=None)

	return x


def trash_cnn_cifar(x, name, is_training, reuse=False):
	# The network is built based on 'NCHW'.
	bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}

	with tf.variable_scope(name, reuse=reuse):
		# x = gridconv2d(x, scope='Conv1_1', num_outputs=32, kernel_size=[3, 3], stride=1, 
		# 							activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
		# 							normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		# x = gridconv2d(x, scope='Conv1_2', num_outputs=32, kernel_size=[3, 3], stride=1, 
		# 							activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
		# 							normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		# x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')
		x = cl.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv1_1')
		x = cl.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
									scope='Conv1_2')
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = gridconv2d(x, scope='Conv2_1', num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = gridconv2d(x, scope='Conv2_2', num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = gridconv2d(x, scope='Conv2_3', num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = gridconv2d(x, scope='Conv3_1', num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = gridconv2d(x, scope='Conv3_2', num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = gridconv2d(x, scope='Conv3_3', num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = tf.reduce_mean(x, [2, 3])
		x = cl.fully_connected(x, num_outputs=10, activation_fn=None)

	return x

# # # # # # # # # CIFAR RESNET # # # # # # # # #
def residual(name, l, is_training, increase_dim=False, first=False):
	bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}
	shape = l.get_shape().as_list()
	in_channel = shape[1]

	if increase_dim:
		out_channel = in_channel * 2
		stride1 = 2
	else:
		out_channel = in_channel
		stride1 = 1

	with tf.variable_scope(name) as scope:
		b1 = l if first else tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
		c1 = cl.conv2d(b1, num_outputs=out_channel, kernel_size=[3, 3], stride=stride1, 
										activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
										normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
										scope='conv1')
		c2 = cl.conv2d(c1, num_outputs=out_channel, kernel_size=[3, 3], stride=1, 
										activation_fn=None, padding='SAME', data_format='NCHW',
										scope='conv2')

		if increase_dim:
			l = cl.avg_pool2d(l, kernel_size=2, stride=2, data_format='NCHW')
			l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

		l = c2 + l
		return l

def grid_residual(name, l, is_training, increase_dim=False, first=False, one_c=False):
	bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}
	shape = l.get_shape().as_list()
	in_channel = shape[1]

	if increase_dim:
		out_channel = in_channel * 2
		stride1 = 2
	else:
		out_channel = in_channel
		stride1 = 1

	with tf.variable_scope(name) as scope:
		b1 = l if first else tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
		c1 = gridconv2d(b1, scope='conv1', num_outputs=out_channel, kernel_size=[3, 3], stride=stride1, 
										activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW', one_c=one_c,
										normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		c2 = gridconv2d(c1, scope='conv2', num_outputs=out_channel, kernel_size=[3, 3], stride=1, 
										activation_fn=None, padding='SAME', data_format='NCHW', one_c=one_c,
										normalizer_fn=None, normalizer_params=None)
		if increase_dim:
			l = cl.avg_pool2d(l, kernel_size=2, stride=2, data_format='NCHW')
			l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

		l = c2 + l
		return l

def resnet(name, n):
	def cnn(x, is_training):
		bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}
		with tf.variable_scope(name) as scope:
			l = cl.conv2d(x, num_outputs=16, kernel_size=[3, 3], stride=1, 
										activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
										normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
										scope='conv0')
			l = tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
			l = residual('res1.0', l, is_training, first=True)
			for k in range(1, n):
				l = residual('res1.{}'.format(k), l, is_training)
			# 32,c=16

			l = grid_residual('res2.0', l, is_training, increase_dim=True)
			for k in range(1, n):
				l = grid_residual('res2.{}'.format(k), l, is_training)
			# 16,c=32

			l = grid_residual('res3.0', l, is_training, increase_dim=True)
			for k in range(1, n):
				l = grid_residual('res3.' + str(k), l, is_training)
			l = tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
			# 8,c=64
			l = tf.reduce_mean(l, [2, 3])

			l = cl.fully_connected(l, num_outputs=10, activation_fn=None)

			return l
	return cnn

def resnet(name, n, grid=False):
	def cnn(x, is_training):
		bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}
		with tf.variable_scope(name) as scope:
			l = cl.conv2d(x, num_outputs=16, kernel_size=[3, 3], stride=1, 
										activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
										normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
										scope='conv0')
			l = tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
			l = residual('res1.0', l, is_training, first=True)
			for k in range(1, n):
				l = residual('res1.{}'.format(k), l, is_training)
			# 32,c=16

			l = residual('res2.0', l, is_training, increase_dim=True)
			for k in range(1, n):
				l = residual('res2.{}'.format(k), l, is_training)
			# 16,c=32

			l = residual('res3.0', l, is_training, increase_dim=True)
			for k in range(1, n):
				l = residual('res3.' + str(k), l, is_training)
			l = tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
			# 8,c=64
			l = tf.reduce_mean(l, [2, 3])

			l = cl.fully_connected(l, num_outputs=10, activation_fn=None)

			return l

	def gridcnn_c(x, is_training):
		bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}
		with tf.variable_scope(name) as scope:
			l = cl.conv2d(x, num_outputs=16, kernel_size=[3, 3], stride=1, 
										activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
										normalizer_fn=cl.batch_norm, normalizer_params=bn_params,
										scope='conv0')
			l = tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
			l = residual('res1.0', l, is_training, first=True)
			for k in range(1, n):
				l = residual('res1.{}'.format(k), l, is_training)
			# 32,c=16

			l = grid_residual('res2.0', l, is_training, increase_dim=True, one_c=True)
			for k in range(1, n):
				l = grid_residual('res2.{}'.format(k), l, is_training, one_c=True)
			# 16,c=32

			l = grid_residual('res3.0', l, is_training, increase_dim=True, one_c=True)
			for k in range(1, n):
				l = grid_residual('res3.{}'.format(k), l, is_training, one_c=True)
			l = tf.nn.relu(cl.batch_norm(l, is_training=is_training, fused=True, data_format='NCHW'))
			# 8,c=64
			l = tf.reduce_mean(l, [2, 3])

			l = cl.fully_connected(l, num_outputs=10, activation_fn=None)

			return l
	return gridcnn_c if grid else cnn

