from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import json

f = open('categories.json', 'r')
categoriesJsonData = json.load(f)
try:
	categories = categoriesJsonData["categories"]
except:
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
				"senior_men_4", "senior_men_5","test"]
f.close()

nb_classes = len(categories)

def main():
    X_train, X_test, y_train, y_test = np.load("./data/face.npy")
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float")  / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
	border_mode='same',
	input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])
    return model

def model_train(X, y):
    model = build_model(X.shape[1:])
    history = model.fit(X, y, batch_size=32, nb_epoch=300, validation_split=0.1)
    hdf5_file = "./data/face-model.h5"
    model.save_weights(hdf5_file)
    return model

def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])

if __name__ == "__main__":
    main()
