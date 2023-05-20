from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
#with open('/home/anup/flaskapi/app/model.json', 'r') as json_file:
    #global loaded_model_json 
    #loaded_model_json = json_file.read()
#model = model_from_json(loaded_model_json)
#model.load_weights('/home/anup/flaskapi/app/weights.h5')
#model.init()

import torch_utils as load_model
global model, graph
model, graph = load_model.init()
model=load_model("/home/anup/flaskapi/app/model.json")
from skimage import io

def generate_input_image(img):
    
    im_resized = img.resize((480, 480))
    X = im_resized.reshape(1, 480, 480, 3)
    print(X)
    X = X/255.0
    return X

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            formFile = request.files['file']
            img = io.imread(formFile)
            X = generate_input_image(img)
            with graph.as_default():
                prediction = model.predict(X)
                print(prediction)
                pred_value = prediction.flatten().tolist()[0]
                return jsonify({ "prediction": pred_value })
        except e:
            print(e)
            return jsonify({'error': 'error during prediction'})
        
    return jsonify({'result' : 1 })

if __name__ == '__main__':
    app.run(debug=True)
