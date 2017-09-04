import numpy as np
import matplotlib.pyplot as plt
import cv2, itertools, os

npy_dir = 'npy/trash_cnn_cifar/'
fig_dir = npy_dir+'fig/'
if not os.path.exists(fig_dir): os.makedirs(fig_dir)

epoch_step = '39_499_'
w = np.load(npy_dir + epoch_step + 'w.npy')
h = np.load(npy_dir + epoch_step + 'h.npy')
i = np.load(npy_dir + epoch_step + 'i.npy')

dim = w.shape[-1]
for img in range(w.shape[0]):
	w_0 = w[img][1]
	h_0 = h[img][1]

	wh_0 = np.array([p for p in itertools.product(w_0, h_0)])

	fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
	ax = fig.gca()
	ax.set_aspect('equal')
	ax.set_xlim([0, dim])
	ax.set_ylim([0, dim])
	ax.set_xticks(np.arange(0, dim, 1))
	ax.set_yticks(np.arange(0, dim, 1))
	plt.scatter(wh_0[:,0], wh_0[:,1], zorder=1, alpha=0.5, c='r', s=10*32/dim)
	plt.imshow(np.transpose(i[img], (1,2,0)), zorder=0, extent=[0, dim, 0, dim])
	#plt.grid()
	fig.savefig(fig_dir+epoch_step+str(img)+'_1.png')
	plt.cla()