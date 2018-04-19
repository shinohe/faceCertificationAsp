from sklearn import cross_validation
from keras.preprocessing.image import load_img, img_to_array
import os, glob
import numpy as np
from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

def createLabelData():
	root_dir = "./image/"
	categories = os.listdir(root_dir)
	nb_classes = len(categories)
	image_size = 32

	X = []
	Y = []
	for idx, cat in enumerate(categories):
		files = glob.glob(root_dir + "/" + cat + "/*")
		if os.path.isdir(os.path.join(root_dir,cat)):
			logger.debug("---%sを処理中" % cat)
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

if __name__ == '__main__':
	createLabelData()