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
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.conv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = cl.conv2d(x, num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.conv2d(x, num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = cl.conv2d(x, num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.conv2d(x, num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = tf.reduce_mean(x, [2, 3])
		x = cl.fully_connected(x, num_outputs=10, activation_fn=None)

	return x


def trash_cnn_cifar(x, name, is_training, reuse=False):
	# The network is built based on 'NCHW'.
	bn_params = {'is_training':is_training, 'fused': True, 'data_format': 'NCHW'}

	with tf.variable_scope(name, reuse=reuse):
		x = cl.gridconv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.gridconv2d(x, num_outputs=32, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = cl.gridconv2d(x, num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.gridconv2d(x, num_outputs=64, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.max_pool2d(x, kernel_size=3, stride=2, data_format='NCHW')

		x = cl.gridconv2d(x, num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = cl.gridconv2d(x, num_outputs=128, kernel_size=[3, 3], stride=1, 
									activation_fn=tf.nn.relu, padding='SAME', data_format='NCHW',
									normalizer_fn=cl.batch_norm, normalizer_params=bn_params)
		x = tf.reduce_mean(x, [2, 3])
		x = cl.fully_connected(x, num_outputs=10, activation_fn=None)

	return x


