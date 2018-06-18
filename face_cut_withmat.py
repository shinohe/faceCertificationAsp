import os
import glob
import argparse
import cv2
import scipy
from PIL import Image
import numpy as np
from mat_read import getExtractData
from mat_read import extractAgeData
from cifar_read import getAllCifarTrainData
from tqdm import tqdm

from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

CASCADE_PATH = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(CASCADE_PATH)
color = (255, 255, 255)

IMAGE_SIZE = 32

def detectFace(image):

	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.07, minNeighbors=9, minSize=(10, 10))

	return facerect

def resize(image):
	return cv2.resize(image, (IMAGE_SIZE ,IMAGE_SIZE))

def rotate(image, r):
	h, w, ch = image.shape # size of image array
	M = cv2.getRotationMatrix2D((w/2, h/2), r, 1) # rotation mat for rotate around image
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated
	
def createOptimizeImage(out_trimming = False):

	# mat file read 
	# wiki
	logger.debug("read wiki mat")
	mat_name = "wiki"
	mat_path = "imdbface/{}_crop/{}.mat".format(mat_name, mat_name)
	extract_age_wiki, extract_gender_wiki, extract_face_score_wiki, extract_full_path, extract_name\
		=getExtractData(mat_path, mat_name)
	
	# imdb
	logger.debug("read imdb mat")
	mat_name = "imdb"
	mat_path = "imdbface/{}_crop/{}.mat".format(mat_name, mat_name)
	extract_age_imdb, extract_gender_imdb, extract_face_score_imdb, extract_full_path_imdb, extract_name_imdb\
		=getExtractData(mat_path, mat_name)
	
	# wiki&imdb
	age_merge = np.concatenate([extract_age_imdb, extract_age_wiki], axis=0)
	gender_merge = np.concatenate([extract_gender_imdb, extract_gender_wiki], axis=0)
	face_score_merge = np.concatenate([extract_face_score_imdb, extract_face_score_wiki], axis=0)
	full_path_merge = np.concatenate([extract_full_path_imdb, extract_full_path], axis=0)
	name_merge = np.concatenate([extract_name_imdb, extract_name], axis=0)
	
	# TODO change external args 
	output_path = "imdbface/imdb_wiki_marge.mat"

	out_images = []
	out_genderes = []
	out_ages = []
	out_names = []
	color = (255, 255, 255)

	# cut face all train data
	logger.debug("create positive data")
	for i in tqdm(range(len(full_path_merge))):
#	for i in tqdm(range(1000)):
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
			facerect_list = detectFace(image)

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
				out_names.append(name_merge[i]);

				# revers
				fliped = np.fliplr(croped)
				revers_file_path = trimming_dir+"/%s%08d.jpg" % (filename ,face_cnt)
				if out_trimming:
					cv2.imwrite(revers_file_path ,resize(fliped))
				face_cnt += 1
				
				out_images.append(resize(fliped));
				out_genderes.append(gender_merge[i]);
				out_ages.append(age_merge[i]);
				out_names.append(name_merge[i]);

	# append negative sample
	logger.debug("create negative data")
	negative_images, negative_names = getAllCifarTrainData()
	for i in tqdm(range(len(negative_images))):
#	for i in tqdm(range(1000)):
		out_images.append(negative_images[i]);
		out_genderes.append(-1);
		out_ages.append(-1);
		out_names.append(negative_names[i]);
		cifar_file_path = "./cifar-100-python/trimming"+"/%s.jpg" % (negative_names[i])
		if out_trimming:
			cv2.imwrite(cifar_file_path ,negative_images[i])
	

	output = {"image": np.array(out_images), "gender": np.array(out_genderes), "age": np.array(out_ages), "name":np.array(out_names), "img_size": IMAGE_SIZE}
	scipy.io.savemat(output_path, output)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--out_trimming", action='store_true', help="trimming")
	
	args = parser.parse_args()
	
	createOptimizeImage(args.out_trimming)
	
	