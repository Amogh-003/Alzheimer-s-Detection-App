from flask import Flask, render_template, request
import os
from model_inference import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            prediction = predict_image(file_path)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
