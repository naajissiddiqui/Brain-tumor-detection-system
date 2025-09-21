from flask import Flask, render_template, requestc, redirect, send_from_directory

from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

#create app
app = Flask(__name__)

#load the trained model
model = load_model('models/model.h5')

#class labels
class_labels=['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = "./uploads"


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# helper or predicted fuction
def predict_tumor(image_path):
    IMAGE_SIZE = 128 
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0 #Normalize pixel values
    img = np.expand_dims(img, axis=0)
    

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]

    return class_labels[predicted_class], confidence

# Routes
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        
        if file:
            # save the file
            file_location = os.path.join(app.config('UPLOAD_FOLDER'), file.filename)
            file.save(file_location)

            # predict results 
            result, confidence = predict_tumor(file_location)


    