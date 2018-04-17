from sklearn import cross_validation
from keras.preprocessing.image import load_img, img_to_array
import os, glob
import numpy as np

root_dir = "./image/"
categories = ["boy_0", "boy_1", "boy_2", "boy_3", "boy_4", "boy_5", "boy_6", "boy_7", "boy_8", "boy_9", 
			"boy_10", "boy_11", "boy_12", "boy_13", "boy_14", "boy_15", "boy_16", "boy_17", "female_0", 
			"female_1", "female_2", "female_3", "female_4", "female_5", "female_6", "female_7", "female_8", 
			"female_9", "female_10", "female_11", "female_12", "female_13", "female_14", "female_15", 
			"female_17", "female_18", "girls_0", "girls_1", "girls_2", "girls_3", "girls_4", "girls_5", 
			"girls_6", "girls_7", "girls_8", "girls_9", "girls_10", "girls_11", "girls_12", "girls_13", 
			"girls_14", "male_0", "male_1", "male_2", "male_3", "male_4", "male_5", "male_6", "male_7", 
			"male_8", "male_9", "senior_female_0", "senior_female_1", "senior_female_2", "senior_female_3", 
			"senior_female_4", "senior_female_5", "senior_female_6", "senior_female_7", "senior_female_8", 
			"senior_female_9", "senior_female_10", "senior_men_0", "senior_men_1", "senior_men_2", "senior_men_3", 
			"senior_men_4", "senior_men_5", "test"]
nb_classes = len(categories)
image_size = 32

X = []
Y = []
for idx, cat in enumerate(categories):
    files = glob.glob(root_dir + "/" + cat + "/*")
    print("---", cat, "を処理中")
    for i, f in enumerate(files):
        img = load_img(f, target_size=(image_size,image_size))
        data = img_to_array(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./data/face.npy", xy)
print("ok,", len(Y))