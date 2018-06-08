from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from mat_read import load_mat
import keras.callbacks
import numpy as np
import json
import os
from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


currentDir = os.path.dirname(os.path.abspath(__file__))
log_filepath = os.path.join(currentDir, "./log")

images, gender, age, image_size = load_mat("imdbface/imdb_wiki_marge.mat")
categorical_size = len(gender)

def main():
	# train data read. if there is not 
	if not os.path.exists("./data/face.npy"):
		X = images
		Y = np.array(range(len(gender)))
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y)
		xy = (X_train, X_test, y_train, y_test)
		np.save("./data/face.npy", xy)
		logger.debug("save data-size : %d" % len(Y))
	else:
		X_train, X_test, y_train, y_test = np.load("./data/face.npy")
	
	X_train = X_train.astype("float") / 256
	X_test  = X_test.astype("float")  / 256
	y_train = np_utils.to_categorical(y_train, categorical_size)
	y_test  = np_utils.to_categorical(y_test, categorical_size)
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
	model.add(Dense(categorical_size))
	model.add(Activation('softmax'))
	model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])
	return model

def model_train(X, y):
	model = build_model(X.shape[1:])
	tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)
	cbks = [tb_cb]
	history = model.fit(X, y, batch_size=32, nb_epoch=150, verbose=1, callbacks=cbks, validation_split=0.1)
	hdf5_file = "./data/face-model.h5"
	model.save_weights(hdf5_file)
	return model

def model_eval(model, X, y):
	score = model.evaluate(X, y)
	print('loss=', score[0])
	print('accuracy=', score[1])

if __name__ == "__main__":
	main()
