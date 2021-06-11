import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img
from flask import Flask
from flask import render_template, request, redirect, flash, url_for
import os




def predict(image):
    vgg_ct = load_model('Models/vgg_ct.h5')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    vgg_pred = vgg_ct.predict(image)
    probability = vgg_pred[0]
    return probability
   


app = Flask(__name__, template_folder='templates')


@app.route('/')
def home_endpoint():
    return redirect('/upload-image')


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            filename = image.filename
            filename = filename.lower()
            jpg = filename.find('jpg')
            jpeg = filename.find('jpeg')
            png = filename.find('png')
            if(jpg==-1 and jpeg==-1 and png==-1):
                flash('Image format should be "png", "jpg" or "jpeg"')
                return redirect(url_for('home_endpoint'))
            answer = predict(image)
            print("Image saved")
            if(answer==0):
                return render_template("0.html")
            else:
                return render_template("1.html")
    return render_template("upload.html")



if __name__ == '__main__':
    # load_model()
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
    app.debug = True
    app.config["IMAGE_UPLOADS"] = "Test/class1"
    app.run(threaded=False,debug=False)
