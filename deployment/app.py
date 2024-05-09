import keras.utils
import numpy as np
from flask import Flask, request, render_template
import os
import FireDetectionModel

app = Flask(__name__, template_folder = ".")
print(os.getcwd())
#model = load_model('../results/V6 Model/V6_model-003-0.967551-0.960252.keras', compile = False)
model = FireDetectionModel.FireDetectionModel(input_shape = (128, 128, 3), use_resnet = True)
model.built= True
model.load_weights('../results/ResNetModel/ResnetV1_model-003-0.975870-0.969558.keras')
#model.load_weights("/home/Minou2105/mysite/ResnetV1_model-003-0.975870-0.969558.keras")

UPLOAD_FOLDER = "/tmp"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def predict():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok = True)
    img_file = request.files["filename"]
    img_file.save(os.path.join(app.config["UPLOAD_FOLDER"], img_file.filename))
    try:
        img = keras.utils.load_img(os.path.join(app.config["UPLOAD_FOLDER"], img_file.filename), target_size=(128,128))
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #img = image.load_img(img_file.filename, target_size=(128, 128))
        print(type(x))
        output = model.predict(x)
        print(output)
        output = model.decode_prediction(output)
        print(output)
    finally:
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], img_file.filename))
    return render_template('index.html', result='Prediction is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
