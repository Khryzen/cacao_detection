import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.preprocessing import image

app = Flask(__name__)

# Load models
leaves_model = tf.keras.models.load_model('leaves.h5')
fruits_model = tf.keras.models.load_model('fruits.h5')

# Labels for the models
leaves_labels = ["Disease", "Healthy"]
fruits_labels = ["Black Rot", "Healthy", "Pod Borer"]


def preprocess_image(base64_str):
    """Preprocess the base64 encoded image."""
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)
    # Convert to PIL Image
    img = Image.open(io.BytesIO(image_data))
    # Resize image to target size
    img = img.resize((224, 224))
    # Convert image to array
    img_array = image.img_to_array(img)
    # Expand dims to match model input
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/leaves', methods=['POST'])
def predict_leaves():
    data = request.get_json(force=True)
    img_base64 = data['image']
    img_array = preprocess_image(img_base64)

    # Predict the class probabilities
    predictions = leaves_model.predict(img_array)
    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    response = {
        "predicted_class": leaves_labels[predicted_class],
        "class_probabilities": predictions.tolist()
    }

    return jsonify(response)


@app.route('/fruits', methods=['POST'])
def predict_fruits():
    data = request.get_json(force=True)
    img_base64 = data['image']
    img_array = preprocess_image(img_base64)

    # Predict the class probabilities
    predictions = fruits_model.predict(img_array)
    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    response = {
        "predicted_class": fruits_labels[predicted_class],
        "class_probabilities": predictions.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
