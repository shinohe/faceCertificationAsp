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
import ssl
import re
import face_makedata
import threading

from mat_read import load_mat
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

# 認証結果のラベル一覧
images, gender, age, image_size = load_mat("imdbface/imdb_wiki_marge.mat")

# flaskでpredict呼べない問題
global model, graph
graph = tf.get_default_graph()
model = face.build_model((image_size.astype("float"), image_size.astype("float"), 3))
model.load_weights("./data/face-model.h5")




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
	

class TrainingThread(threading.Thread):
	progress = 0
	statusDisplay = ""
	
	def __init__(self):
		super(TrainingThread, self).__init__()
		self.stop_event = threading.Event()
	
	def stop(self):
		self.stop_event.set()
	
	def run(self):
		try:
			statusDisplay = "ラベルデータの作成中..."
			logger.info("ラベルデータの作成中...");
			if os.path.exists(os.path.join("data", "face.npy")):
				os.remove(os.path.join("data", "face.npy"))
			face_makedata.createLabelData()
			
			progress = 4
			
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
			
			progress = 5
			statusDisplay = "ラベルデータの学習中..."
			logger.info("ラベルデータの学習中...");
			
			face.main();
			
		finally:
			logger.info("face training finish!!");

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
	page = 1
	# index.html をレンダリングする
	return render_template('index.html', page=page, isMenu=True)

@app.route('/sp/')
def indexSp():
	page = 1
	# index.html をレンダリングする
	return render_template('sp.html', page=page, isMenu=True)

@app.route('/demo')
def demo():
	page = 1
	# demo.html をレンダリングする
	return render_template('demo.html', page=page, isMenu=True)

@app.route('/demoCertification')
def demoCertification():
	page = 1
	# demoCertification.html をレンダリングする
	return render_template('demoCertification.html', page=page, isMenu=True)

@app.route('/labelManager')
def labelManager():
	page = 1
	# labelManager.html をレンダリングする
	return render_template('labelManager.html', page=page, isMenu=True)
	
@app.route('/faceDetection', methods=['GET'])
def faceDetection():
	# faceDetection.html をレンダリングする
	return render_template('faceDetection.html', isMenu=True)


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
	
	return jsonify({'predict':predict,'categories':{'gender':gender, 'age':age}})


@app.route('/allTrainImage', methods=['POST'])
def allTrainImage():
	# すべての学習データ一覧を取得する
	# データ構造 (今はjpgだけ対応しているので、そのうちほかのも対応する)
	# image --
	#        |-label1--
	#                 |-image1.jpg
	#                 |-image2.jpg
	#        |-label2--
	#                 |-image1.jpg
	#        |-etc...
	path = os.path.join(currentDir, 'image')
	allTrainList = {}
	dirs = os.listdir(path)
	for dir in dirs:
		fileList = os.listdir(os.path.join(path, dir))
		allTrainList[dir] = fileList
	
	return jsonify(allTrainList)
	
@app.route('/createLabel', methods=['POST'])
def createLabel():
	path = os.path.join(currentDir, 'image')
	if request.data:
		content_body_dict = json.loads(request.data)
		
		if 'labelName' in content_body_dict:
			labelName = request.json.get('labelName')
			labelName = re.sub(r'[\\|/|:|?|.|"|<|>|\|]', '-', labelName)
			os.makedirs(os.path.join(path,labelName))
			return allTrainImage()
	
	return jsonify({'status':'ERROR','error_message':'不正なリクエストです。'})
	
@app.route('/deleteImage', methods=['POST'])
def deleteImage():
	path = os.path.join(currentDir, 'image')
	if request.data:
		content_body_dict = json.loads(request.data)
		
		if 'labelName' in content_body_dict:
			labelName = request.json.get('labelName')
			path = os.path.join(path, labelName)
		else :
			return jsonify({'status':'ERROR','error_message':'不正なリクエストです。'})
		
		if 'imageName' in content_body_dict:
			imageName = request.json.get('imageName')
			path = os.path.join(path, imageName)
			os.remove(path)
			return allTrainImage()
		else :
			return jsonify({'status':'ERROR','error_message':'不正なリクエストです。'})
	
	return jsonify({'status':'ERROR','error_message':'不正なリクエストです。'})

@app.route('/haarcascade_frontalface_alt.xml', methods=['GET'])
def cascadeFile():
	content = get_file('haarcascade_frontalface_alt.xml')
	return Response(content, mimetype='auto')

@app.route('/sp/haarcascade_frontalface_alt.xml', methods=['GET'])
def cascadeSpFile():
	content = get_file('haarcascade_frontalface_alt.xml')
	return Response(content, mimetype='auto')

@app.errorhandler(InvalidUsage)
def error_handler(error):
	response = jsonify({ 'message': error.message, 'result': error.status_code })
	return response, error.status_code

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('server.crt', 'server.key', 'kurobuta1')

if __name__ == '__main__':
        app.debug = True  #デバッグモード有効化
        app.run(host='0.0.0.0', port=5001, threaded=True) # どこからでもアクセス可能に
#        app.run(host='0.0.0.0', port=5001, ssl_context=context, threaded=True) # どこからでもアクセス可能に
