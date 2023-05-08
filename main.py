from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse
import tensorflow as tf
from skimage import io
import cv2
import requests
import numpy as np

app = Flask(__name__)

cifar10model = tf.keras.models.load_model('cifar10.h5')

cifar10Classes = ['avion ✈️', 'voiture 🚓', 'oiseau 🐦', 'chat 🐱', 'cerf 🦌', 'chien 🐶', 'grenouille 🐸', 'cheval 🐎', 'bateau 🚢', 'camion 🚛']

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            url = request.form.get('url')
            if url is not None:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
                imageFromUrl = requests.get(url, headers=headers)
                image_array = np.array(bytearray(imageFromUrl.content), dtype=np.uint8)
                prediction = predictionFromImageCifar10(image_array)
                bestPrediction = prediction[0][np.argmax(prediction[0])]
                if bestPrediction < 0.5:
                    return jsonify(predictionImageUrl="TODO", predictionText="Le modèle n'a pas pu classifier cette image")

                result = "Prediction: " + cifar10Classes[np.argmax(prediction[0])]
                return jsonify(predictionImageUrl="", predictionText=result)
            if request.is_json:
                parsed_url = urlparse(request.get_json()["imageTest"])
                # TODO parsed_url.path
            else:
                return "" 
        else:
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return ""
            if file:
                file.save('static/uploads/' + file.filename)
    return jsonify(predictionImageUrl="TODO", predictionText="TODO")

@app.route("/")
def index():
    return render_template('index.html')

def predictionFromImageCifar10(image_array):
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = (np.expand_dims(img,0))
    prediction = cifar10model.predict(img)
    return prediction


cocomodel = tf.keras.models.load_model('cifar10.h5') # TODO

cocoClasses = ['avion ✈️', 'voiture 🚓', 'oiseau 🐦', 'chat 🐱', 'cerf 🦌', 'chien 🐶', 'grenouille 🐸', 'cheval 🐎', 'bateau 🚢', 'camion 🚛'] # TODO
@app.route('/upload_coco', methods=['POST'])
def upload_image_coco():
    if request.method == 'POST':
        if 'file' not in request.files:
            url = request.form.get('url')
            if url is not None:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
                imageFromUrl = requests.get(url, headers=headers)
                image_array = np.array(bytearray(imageFromUrl.content), dtype=np.uint8)
                prediction = predictionFromImageCoco(image_array)
                bestPrediction = prediction[0][np.argmax(prediction[0])]
                if bestPrediction < 0.5:
                    return jsonify(predictionImageUrl="TODO", predictionText="Le modèle n'a pas pu classifier cette image")

                result = "Prediction: " + cocoClasses[np.argmax(prediction[0])]
                return jsonify(predictionImageUrl="", predictionText=result)
            if request.is_json:
                parsed_url = urlparse(request.get_json()["imageTest"])
                # TODO parsed_url.path
            else:
                return "" 
        else:
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return ""
            if file:
                file.save('static/uploads/' + file.filename)
    # model.predict()
    return jsonify(predictionImageUrl="TODO", predictionText="TODO")

def predictionFromImageCoco(image_array):
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32)) # TODO resize
    img = (np.expand_dims(img,0))
    prediction = cocomodel.predict(img)
    return prediction