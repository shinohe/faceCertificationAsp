import os
import glob
import argparse
import cv2
import scipy
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

IMAGE_SIZE = 64

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
#	logger.debug( "RESIZED: %s[%sx%s] --> %sx%s" % (filename, before_x, before_y, x, y) )


def resize(image):
	return cv2.resize(image, (IMAGE_SIZE ,IMAGE_SIZE))

def rotate(image, r):
	h, w, ch = image.shape # size of image array
	M = cv2.getRotationMatrix2D((w/2, h/2), r, 1) # rotation mat for rotate around image
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated
	
def create_optimize_image(out_trimming = False):

	# mat file read 
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
	
	
	# TODO let it be external args 
	output_path = "imdbface/imdb_wiki_marge.mat"

	out_images = []
	out_genderes = []
	out_ages = []
	color = (255, 255, 255)

	# cut face all train data
	for i in tqdm(range(len(full_path_merge))):
#	for i in tqdm(range(50)):
		face_cnt = 0

		# get jpg file 
		file_path = full_path_merge[i]
		filename = os.path.basename(file_path)
		file_dir = os.path.dirname(file_path)
		
		image = cv2.imread(file_path)
		if image is None:
			continue

		trimming_dir = file_dir + "/_trimming"
		if out_trimming and not os.path.exists(trimming_dir):
			os.makedirs(trimming_dir)

		for r in range(-12,13,4):
			# rotate 
			image = rotate(image, r)
			# face detect
			facerect_list = detect_face(image)

			# detect face size > 0 
			if len(facerect_list) == 0:
				continue
		
			for rect in facerect_list:
				# save 
				croped = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
				croped_file_path = trimming_dir+"/%s%08d.jpg" % (filename ,face_cnt)
				if out_trimming:
					cv2.imwrite(croped_file_path , resize(croped))
				face_cnt += 1
				
				out_images.append(resize(croped));
				out_genderes.append(gender_merge[i]);
				out_ages.append(age_merge[i]);

				# revers
				fliped = np.fliplr(croped)
				revers_file_path = trimming_dir+"/%s%08d.jpg" % (filename ,face_cnt)
				if out_trimming:
					cv2.imwrite(revers_file_path ,resize(fliped))
				face_cnt += 1
				
				out_images.append(resize(fliped));
				out_genderes.append(gender_merge[i]);
				out_ages.append(age_merge[i]);


	output = {"image": np.array(out_images), "gender": np.array(out_genderes), "age": np.array(out_ages), "img_size": IMAGE_SIZE}
	scipy.io.savemat(output_path, output)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--out_trimming", action='store_true', help="trimming")
	
	args = parser.parse_args()
	
	create_optimize_image(args.out_trimming)
	
	