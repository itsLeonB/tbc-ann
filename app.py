from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import model_from_json
import mahotas as mh
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

class_dict = {'Tubercolosis': 0, 'Normal': 1}

IMM_SIZE = 224

def predict_label(img_path):
    img = mh.imread(img_path)
    
    if len(img.shape) > 2:
        img = mh.resize_to(img, [IMM_SIZE, IMM_SIZE, img.shape[2]])
    else:
        img = mh.resize_to(img, [IMM_SIZE, IMM_SIZE]) 
    if len(img.shape) > 2:
        img = mh.colors.rgb2grey(img[:,:,:3], dtype = np.uint8)
        
    img = np.array(img) / 255
    img = img.reshape(-1, IMM_SIZE, IMM_SIZE, 1)
    pred = model.predict(img)
    pred = np.argmax(pred, axis=1)
    pred = pred.reshape(1, -1)[0]
    diag = {i for i in class_dict if class_dict[i] == pred}
    
    return diag

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)