from decorator import decorator
import tensorflow as tf
import tensorflow.contrib.layers as cl

@decorator
def log_weights(f, *args, **kwargs):
	x = f(*args, **kwargs)
	for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
		print (v)

	return x