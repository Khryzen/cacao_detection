from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

from keras.preprocessing import image
leaves = tf.keras.models.load_model('leaves.h5')
leaves_labels = ["Disease", "Healthy"]

fruits = tf.keras.models.load_model('fruits.h5')
fruits_labels = ["Black Rot", "Healthy", "Pod Borer"]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure the upload folder
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set the allowed extensions (optional)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/leaves', methods=['POST'])
def predict_leaves():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_path = file_path
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)
        print("Preprocessed Image Shape:", img_array.shape)

        # Predict the class probabilities
        predictions = leaves.predict(img_array)
        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        # Print the predicted class and corresponding probabilities
        print("Predicted Class:", leaves_labels[predicted_class])
        print("Class Probabilities:", predictions)
        # Here you can add additional processing for the image
        return jsonify({'message': 'File successfully uploaded', 'prediction': leaves_labels[predicted_class]}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/fruits', methods=['POST'])
def predict_fruits():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_path = file_path
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)
        print("Preprocessed Image Shape:", img_array.shape)

        # Predict the class probabilities
        predictions = fruits.predict(img_array)
        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        # Print the predicted class and corresponding probabilities
        print("Predicted Class:", fruits_labels[predicted_class])
        print("Class Probabilities:", predictions)
        # Here you can add additional processing for the image
        return jsonify({'message': 'File successfully uploaded', 'prediction': fruits_labels[predicted_class]}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
