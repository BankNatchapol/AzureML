from flask import Flask, render_template, request
from io import BytesIO
import os
from PIL import Image
import numpy as np
from base64 import b64encode

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

# code which helps initialize our server
app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
bootstrap = Bootstrap(app)
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

from skorch.classifier import NeuralNetClassifier 

from skimage.filters import threshold_otsu
from skimage.util import invert, img_as_ubyte, img_as_bool
from skimage.morphology import  binary_closing, thin

from sklearn.metrics import  mean_absolute_error
import cv2

criterion = nn.CrossEntropyLoss

googlenet = models.googlenet(pretrained=True)

saved_model = NeuralNetClassifier(module=googlenet,criterion = criterion).initialize()
                            
saved_model.load_params(f_params='googlenetCheckPoint.pt')

class Erode(object):
    def __init__(self):
        self.kernel = np.ones((5,5),np.uint8)

    def __call__(self, sample):
        image = sample[0].numpy()
        #print(image.shape)
        erosion = cv2.erode(image,self.kernel,iterations = 1)
        return  torch.from_numpy(erosion)

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img<tol
    return img[np.ix_(mask.any(1),mask.any(0))]

class Crop(object):
  def __call__(self, sample):
        image = sample.numpy()
        croped = crop_image(image, 1)
        return  torch.from_numpy(croped[None,None,:,:])

def get_binary(img):    
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary

def to_thin(fn):    
    im = img_as_bool(get_binary(invert(fn)))
    out = binary_closing(thin(im)).astype(np.float32)
    return out[None,:,:]

transform = transforms.Compose([
                                  transforms.Grayscale(num_output_channels=1),  
                                  transforms.ToTensor(),
                                  Erode(),
                                  Crop(),
                                  transforms.Lambda(lambda x: F.interpolate(x, size=(128,128))),
                                  transforms.Lambda(lambda x:  to_thin(np.array(x[0][0]))),
                                  transforms.Lambda(lambda x: np.repeat(x,3, axis=0)),
                                  ]) 

class UploadForm(FlaskForm):
    photo = FileField('Upload an image',validators=[FileAllowed(['jpg', 'png', 'jpeg'], u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Predict')

def preprocess(img):
    return transform(img)

@app.route('/', methods=['GET','POST'])
def predict():
    form = UploadForm()
    if form.validate_on_submit():
        print(form.photo.data)
        image_stream = form.photo.data.stream
        original_img = Image.open(image_stream)
        img = preprocess(original_img)
        img = np.expand_dims(img, axis=0)
        prediction = saved_model.predict(img)

        result = prediction[0]
        byteIO = BytesIO()
        original_img.save(byteIO, format=original_img.format)
        byteArr = byteIO.getvalue()
        encoded = b64encode(byteArr)
        print('predict: ',result)
        return render_template('result.html', result=result, encoded_photo=encoded.decode('ascii'))

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)