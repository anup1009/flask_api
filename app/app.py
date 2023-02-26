from flask import Flask, jsonify, request
from torch_utils import transform_image
from flask_cors import CORS, cross_origin
import io
from PIL import Image
from torch_utils import get_prediction

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


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
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            prediction = get_prediction(img)
            data =  {'prediction': prediction}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})
        
    return jsonify({'result' : 1 })

if __name__ == '__main__':
    app.run(debug=True)