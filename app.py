import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os
import cv2
import matplotlib.cm as cm

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model_dump/model.keras")
classes = ['Benign', 'Early Stage', 'Pre-Cancerous', 'Pro-Cancerous']
classes_lower = ['benign', 'early', 'pre', 'pro']

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict_page():
    return render_template("predict.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/upload', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('predict.html', error="No file uploaded")
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('predict.html', error="No file selected")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_index]
        confidence = round(float(prediction[0][predicted_index]) * 100, 2)
        
        all_predictions = []
        for i, prob in enumerate(prediction[0]):
            all_predictions.append({
                'class': classes[i],
                'confidence': round(float(prob) * 100, 1),
                'css_class': classes_lower[i]
            })
        
        return render_template('results.html',
                             predicted_class=predicted_class,
                             confidence=confidence,
                             filename=filename,  # Just filename
                             all_predictions=all_predictions)
    
    return render_template('predict.html', error="Invalid file type")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
