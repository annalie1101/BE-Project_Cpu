import os
import shutil
from flask import Flask, render_template, request
import torch

from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
import main_alpr

CHAR_CLASSIFICATION_WEIGHTS = 'test_data/newweight331.h5'
LP_weights = 'test_data/yolov7_weights_1000imgs_4classes_50epoch.pt'

model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
device = torch.device('cpu')
model_LP = attempt_load(LP_weights, map_location=device)

app = Flask(__name__)  # An instance of the Flask application is created.


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']

    output_folder_path = "static"  # path where the processed images will be stored
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path)

    file.save('static/uploaded_image.jpg')

    license_plate = main_alpr.process_image('static/uploaded_image.jpg', model_LP, model_char, device)

    return render_template('result.html', license_plate=license_plate)

if __name__ == '__main__':
    app.run()
