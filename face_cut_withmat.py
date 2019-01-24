#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import cv2
from PIL import Image
import numpy as np
from mat_read import get_extract_data
from mat_read import extract_age_data
from tqdm import tqdm

from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

IMAGE_HEIGHT = 500
CASCADE_PATH = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(CASCADE_PATH)
color = (255, 255, 255)

def detect_face(image):

	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.07, minNeighbors=9, minSize=(10, 10))

	return facerect

def pre_resize(before, after, height=IMAGE_HEIGHT, filename="", antialias_enable=True):
	"""
	Resize images according to the pre-defined image_heiht regardless of the size of them.
	"""
	
	img = Image.open(before, 'r')
	before_x, before_y = img.size[0], img.size[1]
	x = int(round(float(height / float(before_y) * float(before_x))))
	y = height
	resize_img = img
	if antialias_enable:
		resize_img.thumbnail((x, y), Image.ANTIALIAS)
	else:
		resize_img = resize_img.resize((x, y))

	resize_img.save(os.path.join(after,filename), 'jpeg', quality=100)
	logger.debug( "RESIZED: %s[%sx%s] --> %sx%s" % (filename, before_x, before_y, x, y) )


def resize(image):
	return cv2.resize(image, (64,64))

def rotate(image, r):
	h, w, ch = image.shape # 画像の配列サイズ
	M = cv2.getRotationMatrix2D((w/2, h/2), r, 1) # 画像を中心に回転させるための回転行列
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated
	
def create_optimize_image(directoryPath):

	# matファイル読み込み
	# wiki
	mat_name = "wiki"
	mat_path = "imdbface/{}_crop/{}.mat".format(mat_name, mat_name)
	extract_age_wiki, extract_gender_wiki, extract_face_score_wiki, extract_full_path\
		=get_extract_data(mat_path, mat_name)
	
	# imdb
	mat_name = "imdb"
	mat_path = "imdbface/{}_crop/{}.mat".format(mat_name, mat_name)
	extract_age_imdb, extract_gender_imdb, extract_face_score_imdb, extract_full_path_imdb\
		=get_extract_data(mat_path, mat_name)
	
	# wiki&imdb
	age_merge = np.concatenate([extract_age_imdb, extract_age_wiki], axis=0)
	gender_merge = np.concatenate([extract_gender_imdb, extract_gender_wiki], axis=0)
	face_score_merge = np.concatenate([extract_face_score_imdb, extract_face_score_wiki], axis=0)
	full_path_merge = np.concatenate([extract_full_path_imdb, extract_full_path], axis=0)

	image_merge = []

	# 顔切り出し
	for i in tqdm(range(len(full_path_merge))):
	
		# jpgファイル取得
		file_path = full_path_merge[i]
		
		resize_dir = os.path.dirname(file_path) + "/_resize"
		logger.debug(resize_dir)
		logger.debug(file_path)
		logger.debug(os.path.basename(file_path))
		if not os.path.exists(resize_dir):
			os.makedirs(resize_dir)
		pre_resize(file_path, resize_dir, filename=os.path.basename(file_path))








#	output = {"image": np.array(out_imgs), "gender": np.array(gender_merge), "age": np.array(age_merge), "img_size": img_size}
#	scipy.io.savemat(output_path, output)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='clip face-image from imagefile and do data argumentation.')
	parser.add_argument('-p', required=True, help='set files path.', metavar='imagefile_path')
	args = parser.parse_args()

	# wiki
	wiki_path = os.path.join(args.p, 'wiki_crop')
	create_optimize_image(wiki_path)
	
	# imdb_crop
#	imdb_crop_path = os.path.join(args.p, 'imdb_crop')
#	create_optimize_image(imdb_crop_path)
	