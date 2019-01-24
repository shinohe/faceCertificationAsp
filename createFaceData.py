#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import cv2
import time
import face_makedata
import shutil
import argparse
import re
from logging import getLogger, StreamHandler, DEBUG
import face_cut
import json


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

cascade_path = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)
cam = cv2.VideoCapture(0)
color = (255, 255, 255)

image_size = 32


def main():
	repatter = re.compile(r".*jp*g")
	
	labelName = args.n
	path = "imdbface"+os.path.sep+"imdb_crop"
	os.mkdir(path)
	logger.debug(labelName)
	
	face_cut.createOptimizeImage(path)
	
	# 不要なフォルダの削除
	
	# 元のjpegファイルをすべて削除
	files = os.listdir(path)
	for file_name in files:
		if repatter.match(file_name):
			src = os.path.join(path, file_name)
			logger.debug("remove : %s" % src)
			os.remove(src)
		
	# トリミング後のファイルをラベル直下のディレクトリに移動
	trimmingPath = os.path.join(path, "_trimming")
	files = os.listdir(trimmingPath)
	for file_name in files:
		if repatter.match(file_name):
			src = os.path.join(trimmingPath, file_name)
			dest = os.path.join(path, file_name)
			logger.debug(src+" ⇒ "+dest)
			shutil.move(src, dest)
	
	# 作成時に作ったディレクトリをすべて削除
	shutil.rmtree(os.path.join(path,"_addbox"))
	shutil.rmtree(os.path.join(path,"_resize"))
	shutil.rmtree(os.path.join(path,"_trimming"))
	
	
	# ラベルデータも作るならラベルのデータを作成しcategories.jsonを出力する
	if os.path.exists(os.path.join("data", "face.npy")):
		os.remove(os.path.join("data", "face.npy"))
	face_makedata.createLabelData()
	files = os.listdir("image")
	# ディレクトリ以外は除外
	dump_files = []
	for file_name in files:
		if os.path.isdir(os.path.join("image",file_name)):
			dump_files.append(file_name)
	dump_dict = {"categories":dump_files}

	f = open('categories.json', 'w')
	json.dump(dump_dict,f,indent=4)
	f.close()

if __name__ == '__main__':
	main()