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



    