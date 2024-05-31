import requests
import base64

# Read the image file
image_path = 'leaves/h3.jpg'  # Replace with your image file path


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode('utf-8')
    return base64_str


base64_str = image_to_base64(image_path)

response = requests.post('http://0.0.0.0:5000/leaves',
                         json={'image': base64_str})

if response.status_code == 200:
    result = response.json()
    print("Predicted Class:", result['predicted_class'])
    print("Class Probabilities:", result['class_probabilities'])
else:
    print("Error:", response.json())
