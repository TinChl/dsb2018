import numpy as np
import os
import matplotlib.pyplot as plt
from config_training import config
import pandas as pd


def main():
	data_dir = config['data_path']
	filenames = filenames = os.listdir(data_dir)
	save_dir = config['preprocess_result_path']

	for id in xrange(len(filenames)):
		imgs = os.listdir(os.path.join(data_dir, filenames[id], 'images'))
		img = plt.imread(os.path.join(data_dir, filenames[id], 'images', '%s' % (imgs[0])))
		masks = os.listdir(os.path.join(data_dir, filenames[id], 'masks'))
		img = np.moveaxis(img, -1, 0)
		
		bboxes = []
		for mask in masks:
		    mask = plt.imread(os.path.join(data_dir, filenames[id], 'masks', mask))
		    yy, xx = np.where(mask == 1)
		    len_y = max(1, np.max(yy) - np.min(yy))
		    len_x = max(1, np.max(xx) - np.min(xx))
		    center_y = np.min(yy) + len_y / 2
		    center_x = np.min(xx) + len_x / 2
		    bboxes.append([center_y, center_x, len_y, len_x])
		#     plt.imshow(mask, cmap='gray')
		#     plt.show()
		bboxes = np.array(bboxes)

		np.save(os.path.join(save_dir, '%s_img.npy' % (filenames[id])), img)
		np.save(os.path.join(save_dir, '%s_label.npy' % (filenames[id])), bboxes)
		print 'Finished preprocessing %d/%d: image shape %s' % (id, len(filenames), str(img.shape))


if __name__ == '__main__':
	main()
