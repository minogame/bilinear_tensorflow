import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
import re

import tensorflow as tf
import tensorflow.contrib.layers as cl
from model import trash_cnn_cifar

# Data loading and preprocessing
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.datasets import cifar10
(cifar10_X, cifar10_Y), (X_test, Y_test) = cifar10.load_data()
cifar10_X = np.transpose(cifar10_X, [0,3,1,2])
cifar10_Y = to_categorical(cifar10_Y, 10)
X_test = np.transpose(X_test, [0,3,1,2])
Y_test = to_categorical(Y_test, 10)

cnn = trash_cnn_cifar(name='grid')
# cnn = resnet('resnet', 5, grid=True)
WEIGHT_DECAY = 1e-4
l2 = cl.l2_regularizer(WEIGHT_DECAY)
batch_size = 100

import logging
logging.basicConfig(filename='textlog_c2.log',
														filemode='a',
														format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
														datefmt='%H:%M:%S',
														level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-4s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

with tf.Graph().as_default():
	X = tf.placeholder(shape=(batch_size, 3, 32, 32), dtype=tf.float32)
	Y = tf.placeholder(shape=(batch_size, 10), dtype=tf.float32)
	is_training = tf.placeholder(shape=(), dtype=tf.bool)
	lr = tf.Variable(0.002, name='learning_rate', trainable=False, dtype=tf.float32)
	decay_lr_op = tf.assign(lr, lr/10)

	def aug_image(x):
		x = tf.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]])
		x = tf.random_crop(x, [batch_size, 3, 32, 32])
		x = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x)
		return x

	XX = tf.cond(is_training, lambda: aug_image(X), lambda: X)

	logits = cnn(XX, is_training=is_training)

	clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

	reg_loss_list = []
	for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		if re.search('weights', v.name):
			reg_loss_list.append(l2(v))
			logging.info('Apply {} for {}'.format(l2.__name__, v.name))
	reg_loss = tf.add_n(reg_loss_list) if reg_loss_list else tf.contant(0.0, dtype=tf.float32)

	loss = clf_loss + reg_loss

	coords_check = tf.get_collection('70f92c137c01d89c6477c5ef22411bfe')
	coords_w = coords_check[0][0]
	coords_h = coords_check[0][1]

	grid_weights = []
	for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
		if re.search('GRID', v.name):
			grid_weights.append(v)
	convbn_weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v not in grid_weights]

	opt_conv = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
	opt_grid = tf.train.MomentumOptimizer(learning_rate=lr*5, momentum=0.9, use_nesterov=True)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op_conv = opt_conv.minimize(loss, var_list=convbn_weights)
		train_op_grid = opt_grid.minimize(loss, var_list=grid_weights)

	accuracy = tf.reduce_mean(
			tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)), tf.float32),
			name='acc')

	with tf.name_scope('Loss_summary'):
		tf.summary.scalar(name='clf_loss', tensor=clf_loss)
		tf.summary.scalar(name='reg_loss', tensor=reg_loss)
	merged_all = tf.summary.merge_all()
	grid_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='grid')
	grid_main_list = [v for v in grid_list if not 'Momentum' in v.name]

	saver = tf.train.Saver(var_list=grid_main_list, max_to_keep=None)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		log_dir = 'log/' + cnn.__name__
		npy_dir = 'npy/' + cnn.__name__ + '/'
		if not os.path.exists(npy_dir): os.makedirs(npy_dir)
		saver.restore(sess, tf.train.latest_checkpoint('ckpt/grid_cnn'))

		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()

		for epoch in range(40):
			cifar10_X, cifar10_Y = shuffle(cifar10_X, cifar10_Y)
			avg_acc = 0.
			avg_loss = 0.

			total_batch = int(len(cifar10_X)/batch_size)
			if epoch == 20 or epoch == 30:
				sess.run(decay_lr_op)

			for i in range(total_batch):
				batch_xs, batch_ys = cifar10_X[batch_size*i:batch_size*(i+1)], cifar10_Y[batch_size*i:batch_size*(i+1)]

				if i % 100 == 0:
					merged, _, cost, _cw, _ch = sess.run([merged_all, train_op_grid, clf_loss, coords_w, coords_h], feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
					iters = i + epoch*total_batch
					summary_writer.add_summary(merged, iters)
					summary_writer.add_run_metadata(run_metadata, 'metadata {}'.format(iters), iters)
					logging.info('Epoch: {0:03d} Step: {1:03d} Loss: {2}'.format((epoch+1), i, cost))
					if i % 499 == 0:
						np.save(npy_dir+'{}_{}_w'.format(epoch, i), _cw)
						np.save(npy_dir+'{}_{}_h'.format(epoch, i), _ch)
						np.save(npy_dir+'{}_{}_i'.format(epoch, i), batch_xs)
					# sess.run(train_op_grid, feed_dict={X: batch_xs, Y: batch_ys, is_training: True})

				else:
					# sess.run(train_op_conv, feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
					sess.run(train_op_grid, feed_dict={X: batch_xs, Y: batch_ys, is_training: True})

			for j in range(100):
				batch_xs, batch_ys = X_test[100*j:100*(j+1)], Y_test[100*j:100*(j+1)]
				acc, _loss = sess.run([accuracy, clf_loss], feed_dict={X: batch_xs, Y: batch_ys, is_training: False})

				avg_acc += acc/100
				avg_loss += _loss/100
			logging.info('Acc = {}, Loss = {}'.format(avg_acc, avg_loss))