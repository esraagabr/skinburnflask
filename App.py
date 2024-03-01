from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from decimal import Decimal

# Define a flask app
app = Flask(__name__)

def preprocess_image(img_path):
    # Reading and Resizing the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

    # Convert the image to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # TensorFlow Lite models expect input tensors as 'float32'
    img_array = img_array.astype(np.float32)

    # Return the image
    return img_array

def model_predict(model_path, img_path):
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    img = preprocess_image(img_path)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    class_probability = interpreter.get_tensor(output_details[0]['index'])

    # Return the Probability of classes
    return class_probability

# Load the model
model_path = r'best_model_eff.tflite'

# Define thresholds for each class
thresholds = [Decimal('0.9996'), Decimal('0.99'), Decimal('0.999')]

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make probability for each class
        probabilities = model_predict(model_path, file_path)

        # Predict the class name based on the thresholds
        predicted_classes = []
        class_labels = ['1st degree burn', '2nd degree burn', '3rd degree burn']
        for i, prob in enumerate(probabilities[0]):
            if prob > thresholds[i]:
                predicted_classes.append(class_labels[i])

        if predicted_classes:
            return ', '.join(predicted_classes)
        else:
            return "No burn detected"

if __name__ == '__main__':
    app.run()
