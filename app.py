import glob
import os
import shutil
from flask import Flask, render_template, request
import torch

from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
import main_alpr

CHAR_CLASSIFICATION_WEIGHTS = 'test_data/cnn_weights.h5'
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

    output_folder_path = os.path.join(app.root_path, 'static')
    os.makedirs(output_folder_path, exist_ok=True)

    if os.path.exists(output_folder_path):
        file_pattern = os.path.join(output_folder_path, "*.jpg")
        jpg_files = glob.glob(file_pattern)

        for file_path in jpg_files:
            os.remove(file_path)

    file.save('static/uploaded_image.jpg')

    license_plate = main_alpr.process_image('static/uploaded_image.jpg', model_LP, model_char, device)

    return render_template('result.html', license_plate=license_plate)

if __name__ == '__main__':
    app.run()
