#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Flask などの必要なライブラリをインポートする
import face_keras as face
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort, Response
import numpy as np
from flask_httpauth import HTTPDigestAuth
import os
import math
import json
import tensorflow as tf

from dist import dist

# debug用
import cv2


from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

import codecs  
import encodings.utf_8

codecs.register(lambda encoding: utf_8.getregentry()) 

currentDir = os.path.dirname(os.path.abspath(__file__))


# flaskでpredict呼べない問題
global model, graph
graph = tf.get_default_graph()
model = face.build_model((32, 32, 3))
model.load_weights("./data/face-model.h5")


# 認証結果のラベル一覧
categoriesFile = open('categories.json', 'r')
jsonData = json.load(categoriesFile)
if jsonData["categories"]:
	categories = jsonData["categories"]
else:
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
				
categoriesFile.close


# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)
app.register_blueprint(dist.app)
app.config['SECRET_KEY'] = 'secret key here'
auth = HTTPDigestAuth()

users = {
	"jdragon1": "jdragon1"
}

# おまじない
def round(x,d=0):
	p=10**d
	return (x*p*2+1)//2/p

class InvalidUsage(Exception):
	status_code = 400

	def __init__(self, message, status_code=None, payload=None):
		Exception.__init__(self)
		self.message = message
		if status_code is not None:
			self.status_code = status_code
		self.payload = payload
		
	def to_dict(self):
		rv = dict(self.payload or ())
		rv['message'] = self.message
		return rv

def checkFacePredict(data):
	pre = None
	try:
		npData = np.array(data)
		npData.astype("float")
		
		X = []
		X.append(npData)
		X = np.array(X)
		X  = X.astype("float")  / 256
		
		# flaskでpredict呼べない問題
		with graph.as_default():
			pre = model.predict(X)[0].tolist()
		
	except:
		import traceback
		traceback.print_exc()
	
	return pre

def get_file(filename):  # pragma: no cover
	try:
		src = os.path.join(currentDir, filename)
		return open(src).read()
	except IOError as exc:
		return str(exc)

# Digest認証
@auth.get_password
def get_pw(username):
	if username in users:
		return users.get(username)
	return None


# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理
@app.route('/')
def index():
	title = u"トップページ"
	page = 1
	# index.html をレンダリングする
	return render_template('index.html', title=title, page=page, isMenu=True)

@app.route('/checkFace', methods=['POST'])
def checkFace():
	predict = None
	if request.data:
		content_body_dict = json.loads(request.data)
		
		if 'input' in content_body_dict:
			inputData = request.json.get('input')
			npData = np.array(inputData)
			npData.astype("int8")
			# BGRの画像
			cv2.imwrite("upload.png", npData)
			uploadImg = cv2.imread("upload.png")
			uploadImg = cv2.cvtColor(uploadImg, cv2.COLOR_BGR2RGB)
			cv2.imwrite("upload.png", uploadImg)

			predict = checkFacePredict(inputData)
	
	return jsonify({'predict':predict,'categories':categories})


@app.route('/haarcascade_frontalface_alt.xml', methods=['GET'])
def cascadeFile():
	content = get_file('haarcascade_frontalface_alt.xml')
	return Response(content, mimetype='auto')

@app.errorhandler(InvalidUsage)
def error_handler(error):
	response = jsonify({ 'message': error.message, 'result': error.status_code })
	return response, error.status_code

if __name__ == '__main__':
	app.debug = True  #デバッグモード有効化
	app.run(host='0.0.0.0', threaded=True) # どこからでもアクセス可能に

