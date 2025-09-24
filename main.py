from flask import Flask, render_template, request, redirect, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Create app
app = Flask(__name__)

# Load the trained model
model = load_model("models/model.h5", compile=False)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper / prediction function
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]

    if class_labels[predicted_class_index] == "notumor":
        return "No Tumor Detected", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            file_location = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)

            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence*100:.2f}",
                file_path=f"uploads/{file.filename}",
            )

    return render_template("index.html", result=None)

# Route to serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Run app
if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, render_template, request, redirect, send_from_directory

# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import os

# #create app
# app = Flask(__name__)

# #load the trained model
# model = load_model('models/model.h5')

# #class labels
# class_labels=['pituitary', 'glioma', 'notumor', 'meningioma']

# # Define the uploads folder
# UPLOAD_FOLDER = "./uploads"


# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # helper or predicted fuction
# def predict_tumor(image_path):
#     IMAGE_SIZE = 128 
#     img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img) / 255.0 #Normalize pixel values
#     img = np.expand_dims(img, axis=0)
    

#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(prediction, axis=1)[0]
#     confidence_score = np.max(prediction, axis=1)[0]

#     if class_labels[predicted_class_index] == 'notumor':
#         result = "No Tumor Detected", confidence_score
#     else: 
#         return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# # Routes
# @app.route("/", methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['file']
        
#         if file:
#             # save the file
#             file_location = os.path.join(app.config('UPLOAD_FOLDER'), file.filename)
#             file.save(file_location)

#             # predict results 
#             result, confidence = predict_tumor(file_location)

#             # return results along with the image path for display
#             return render_template('index.html', result=result, confidence=f'{confidence*100:.2f  }', file_path=f'uploads/{file.filename}')
    
#     return  render_template('index.html', result=None)

# # Route to serve uploaded files
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config[UPLOAD_FOLDER], filename)

# # python main
# if __name__ =='main':
#   app.run(debug=True)