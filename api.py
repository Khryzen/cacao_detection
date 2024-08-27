# import base64
# import io
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from flask import Flask, request, jsonify
# from keras.preprocessing import image

# app = Flask(__name__)

# # Load models
# leaves_model = tf.keras.models.load_model('leaves.h5')
# fruits_model = tf.keras.models.load_model('fruits.h5')

# # Labels for the models
# leaves_labels = ["Disease", "Healthy"]
# fruits_labels = ["Black Rot", "Healthy", "Pod Borer"]


# def preprocess_image(base64_str):
#     """Preprocess the base64 encoded image."""
#     # Decode the base64 string
#     image_data = base64.b64decode(base64_str)
#     # Convert to PIL Image
#     img = Image.open(io.BytesIO(image_data))
#     # Resize image to target size
#     img = img.resize((224, 224))
#     # Convert image to array
#     img_array = image.img_to_array(img)
#     # Expand dims to match model input
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array


# @app.route('/leaves', methods=['POST'])
# def predict_leaves():
#     data = request.get_json(force=True)
#     # img_base64 = data['image']
#     img_base64 = data.get('image')
#     img_array = preprocess_image(img_base64)

#     # Predict the class probabilities
#     predictions = leaves_model.predict(img_array)
#     # Get the predicted class label
#     predicted_class = np.argmax(predictions)

#     response = {
#         "predicted_class": leaves_labels[predicted_class],
#         "class_probabilities": predictions.tolist()
#     }

#     return jsonify(response)


# @app.route('/fruits', methods=['POST'])
# def predict_fruits():
#     data = request.get_json(force=True)
#     img_base64 = data['image']
#     img_array = preprocess_image(img_base64)

#     # Predict the class probabilities
#     predictions = fruits_model.predict(img_array)
#     # Get the predicted class label
#     predicted_class = np.argmax(predictions)

#     response = {
#         "predicted_class": fruits_labels[predicted_class],
#         "class_probabilities": predictions.tolist()
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0")


import base64
import io
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.preprocessing import image
import requests
from flask_cors import CORS  # Import the CORS extension

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load models
leaves_model = tf.keras.models.load_model('leaves.h5')
fruits_model = tf.keras.models.load_model('fruits.h5')

# Labels for the models
leaves_labels = ["Disease", "Healthy"]
fruits_labels = ["Black Rot", "Healthy", "Pod Borer"]

# Preprocess image function
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

# Predict leaves route
# @app.route('/leaves', methods=['POST'])
# def predict_leaves():
#     data = request.get_json(force=True)
#     img_base64 = data.get('image')
#     img_array = preprocess_image(img_base64)

#     # Predict the class probabilities
#     predictions = leaves_model.predict(img_array)
#     # Get the predicted class label
#     predicted_class = np.argmax(predictions)

#     response = {
#         "predicted_class": leaves_labels[predicted_class],
#         "class_probabilities": predictions.tolist()
#     }

#     return jsonify(response)

@app.route('/leaves', methods=['POST'])
def predict_leaves():
    decode_image_from_url("http://0.0.0.0:8000/media/img/d2.jpg")
    # Get the image URL from form data
    image_url = request.form.get('image_url')

    # Decode image from URL
    img = decode_image_from_url(image_url)
    if img is None:
        return jsonify({"error": "Failed to decode image from URL"}), 400

    # Preprocess the decoded image
    img_array = preprocess_image(img)

    # Predict the class probabilities
    predictions = leaves_model.predict(img_array)
    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    response = {
        "predicted_class": leaves_labels[predicted_class],
        "class_probabilities": predictions.tolist()
    }

    return jsonify(response)

# Predict fruits route
@app.route('/fruits', methods=['POST'])
def predict_fruits():
    data = request.get_json(force=True)
    img_base64 = data.get('image')
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

def decode_image_from_url(url):
  response = requests.get(url)

  # Check if the request was successful (status code 200)
  if response.status_code == 200:
      # Decode the image
      img = Image.open(BytesIO(response.content))
      # Now you can work with the image object
      # For example, you can display the image
      img.show()
      return img
  else:
      print("Failed to fetch the image. Status code:", response.status_code)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
