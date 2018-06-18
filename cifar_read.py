from scipy.io import loadmat
from logging import getLogger, StreamHandler, DEBUG
import numpy as np
import cv2
import _pickle
import os

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

# TODO change external args 

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = _pickle.load(fo, encoding='bytes')
	return dict

def openCifarImageFile(file_path):
	data_dict = unpickle(file_path)
	train_data = data_dict[b'data']
	train_fine_labels = data_dict[b'fine_labels']
	train_coarse_labels = data_dict[b'coarse_labels']
	return train_data, train_fine_labels, train_coarse_labels

def getCifarImageData(train_data):
	np_data = np.array(train_data)
	np_data = np.rollaxis(np_data.reshape((3,32,32)),0,3)
	return np_data


def getAllCifarTrainData(cifer_path = './cifar-100-python'):
	out_images = []
	out_names = []

	train_path = os.path.join(cifer_path, 'train')
	test_path = os.path.join(cifer_path, 'test')

	train_data ,train_fine_labels ,train_coarse_labels = openCifarImageFile(train_path)
	test_data ,test_fine_labels ,test_coarse_labels = openCifarImageFile(train_path)

	meta_dic = unpickle(os.path.join(cifer_path, 'meta'))
	clabel_names = meta_dic[b'coarse_label_names']
	flabel_names = meta_dic[b'fine_label_names']

	roop_count = len(train_fine_labels)
	for i in range(roop_count):
		np_data = getCifarImageData(train_data[i])
		fine_label = flabel_names[train_fine_labels[i]]
		coarse_label = clabel_names[train_coarse_labels[i]]
		
		out_images.append(np_data);
		out_names.append("%s_%s" % (fine_label, coarse_label));
		
	return out_images, out_names

