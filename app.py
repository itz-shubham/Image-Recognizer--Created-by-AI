# This is a web application that can classify images using ResNet50 model.
# Images are uploaded to the server and then classification is performed.
# The results are displayed in a web page.

import os
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained ResNet50 model
model = models.resnet50(weights=None)
model.eval()

# Define the data transformation steps
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('results', filename=filename))
    return render_template('upload.html')

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(filepath).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    import json
    import requests
    response = requests.get(labels_url)
    labels = json.loads(response.text)

    top3_idx = probabilities.topk(3).indices.numpy()
    top3_prob = probabilities.topk(3).values.numpy()
    top3_labels = [labels[str(idx)] for idx in top3_idx]

    recognition_data = []
    top_class_name = None
    for idx, prob in zip(top3_idx, top3_prob):
        label = labels[str(idx)]
        class_name = label[1]  # Human-readable class name
        if top_class_name is None:
            top_class_name = class_name  # Store the top class name

        recognition_data.append({'name': class_name, 'probability': f"{prob * 100:.2f}%" })

    return render_template(
        'results.html', 
        image_url=url_for('uploaded_file', filename=filename),
        recognition_data=recognition_data
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

