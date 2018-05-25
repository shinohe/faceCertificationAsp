"""
Python 3
"""

import os
import glob
import argparse
import cv2
from PIL import Image
import numpy as np

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

	resize_img.save(after, 'jpeg', quality=100)
	logger.debug( "RESIZED: %s[%sx%s] --> %sx%s" % (filename, before_x, before_y, x, y) )


def resize(image):
	return cv2.resize(image, (64,64))

def rotate(image, r):
	h, w, ch = image.shape # 画像の配列サイズ
	M = cv2.getRotationMatrix2D((w/2, h/2), r, 1) # 画像を中心に回転させるための回転行列
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated
	
def create_optimize_image(directoryPath):
	# リサイズした画像を格納
	resize_dir = directoryPath + "/_resize"
	if not os.path.exists(resize_dir):
		os.makedirs(resize_dir)

	# 顔部分に囲いを追加した画像を格納
	addbox_dir = directoryPath + "/_addbox"
	if not os.path.exists(addbox_dir):
		os.makedirs(addbox_dir)

	# 顔部分をトリミングした画像を格納
	trimming_dir = directoryPath + "/_trimming"
	if not os.path.exists(trimming_dir):
		os.makedirs(trimming_dir)
	
	face_cnt = 0

	# jpgファイル取得
	files = glob.glob( "%s/*.jp*g" % (directoryPath) )
	
	# resize
	for file_name in files:
		before_path = file_name
		filename = os.path.basename(file_name)
		after_path = '%s/%s' % ( resize_dir, filename )
		pre_resize(before_path, after_path, filename=file_name)

	resize_files = glob.glob(resize_dir+"/*.jpg")
	
	for file_name in resize_files:
		logger.debug("detect face on file:"+file_name)

		# 画像のロード
		image = cv2.imread(file_name)
		if image is None:
			# 読み込み失敗
			logger.debug("image is None")
			continue

		# -12~12度の範囲で3度ずつ回転
		for r in range(-12,13,4):
			image = rotate(image, r)

			# 顔画像抽出
			facerect_list = detect_face(image)
			if len(facerect_list) == 0:
				continue

			basename = os.path.basename(file_name)

			# 顔検知の囲い追加画像保存 どの程度の精度で検知できているかの確認
			for rect in facerect_list:
				cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)

			cv2.imwrite(addbox_dir+"/"+basename, image)

			# 顔部分切り抜き
			for facerect in facerect_list:
				# 顔画像部分の切り抜き
				croped = image[facerect[1]:facerect[1]+facerect[3],facerect[0]:facerect[0]+facerect[2]]

				# 出力
				cv2.imwrite(trimming_dir+"/%08d.jpg" % face_cnt, resize(croped))
				face_cnt += 1

				# 反転画像も出力
				fliped = np.fliplr(croped)
				cv2.imwrite(trimming_dir+"/%08d.jpg" % face_cnt, resize(fliped))
				face_cnt += 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='clip face-image from imagefile and do data argumentation.')
	parser.add_argument('-p', required=True, help='set files path.', metavar='imagefile_path')
	args = parser.parse_args()
	
	create_optimize_image(args.p)
	