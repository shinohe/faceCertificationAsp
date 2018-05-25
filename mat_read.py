import numpy as np
import os
from scipy.io import loadmat
from tqdm import tqdm
from datetime import datetime

def calc_age(taken, dob):
	birth = datetime.fromordinal(max(int(dob) - 366, 1))

	if birth.month < 7:
		return taken - birth.year
	else:
		return taken - birth.year - 1

def get_meta(mat_path, mat_name):
	meta = loadmat(mat_path)
	mat_dir = os.path.dirname(mat_path)
	full_path = []
	for i in range(len(meta[mat_name][0, 0]["full_path"][0])):
		full_path.append(os.path.join(mat_dir, str(meta[mat_name][0, 0]["full_path"][0][i][0])))
	dob = meta[mat_name][0, 0]["dob"][0]  # Matlab serial date number
	gender = meta[mat_name][0, 0]["gender"][0]
	photo_taken = meta[mat_name][0, 0]["photo_taken"][0]  # year
	face_score = meta[mat_name][0, 0]["face_score"][0]
	second_face_score = meta[mat_name][0, 0]["second_face_score"][0]
	age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

	return full_path, dob, gender, photo_taken, face_score, second_face_score, age

# extract effective data 
def get_extract_data(mat_path, mat_name):
	full_path, dob, gender, photo_taken, face_score, second_face_score, age\
		= get_meta(mat_path, mat_name)
	
	extract_age = []
	extract_gender = []
	extract_face_score = []
	extract_full_path = []
	
	for i in tqdm(range(len(face_score))):
		# face_score >= 1
		if face_score[i] < 1:
			continue
		
		# second_face_score only 0
		if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
			continue
		
		# age between 0 and 100
		if ~(0 <= age[i] <= 100):
			continue
		
		# gender is nan
		if np.isnan(gender[i]):
			continue
		
		extract_age.append(age[i])
		extract_gender.append(gender[i])
		extract_face_score.append(face_score[i])
		extract_full_path.append(full_path[i])

	return extract_age, extract_gender, extract_face_score, extract_full_path

# extract effective data & limit
def extract_age_data(age, gender, face_score, full_path, age_limit_count=1000):
	extract_age = []
	extract_gender = []
	extract_face_score = []
	extract_full_path = []
	age_count={}
	
	for i in tqdm(range(len(age))):
		if age[i] in age_count:
			age_count[age[i]] += 1
		else :
			age_count[age[i]] = 1;
		if age_count[age[i]] > age_limit_count:
			continue
		extract_age.append(age[i])
		extract_gender.append(gender[i])
		extract_face_score.append(face_score[i])
		extract_full_path.append(full_path[i])
		
	return extract_age, extract_gender, extract_face_score, extract_full_path