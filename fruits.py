import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image

from keras.preprocessing import image
best_model = tf.keras.models.load_model('fruits.h5')


labels = ["Black Rot", "Healthy", "Pod Borer"]

img_path = 'fruits/black_pod_rot_2.jpg'
# img_path = 'drive/MyDrive/cacao/cacao_photos/pod_borer/pod_borer_2.jpg'
# img_path = 'drive/MyDrive/cacao/cacao_photos/black_pod_rot/black_pod_rot_3.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
print("Preprocessed Image Shape:", img_array.shape)

# Predict the class probabilities
predictions = best_model.predict(img_array)
# Get the predicted class label
predicted_class = np.argmax(predictions)

# Print the predicted class and corresponding probabilities
print("Predicted Class:", labels[predicted_class])
print("Class Probabilities:", predictions)
