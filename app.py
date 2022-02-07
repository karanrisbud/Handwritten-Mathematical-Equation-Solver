from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import imutils
from imutils.contours import sort_contours
import math
import warnings
import os
warnings.filterwarnings("ignore")
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    validation_split=0.2)
training_set = train_datagen.flow_from_directory('dataset',subset="training",shuffle=True,
    target_size = (64,64),batch_size = 32,class_mode = 'categorical',color_mode="grayscale")

testing_set = train_datagen.flow_from_directory('dataset',subset="validation",shuffle=True,
    target_size = (64,64),batch_size = 32,class_mode = 'categorical',color_mode="grayscale")

cnn = tf.keras.models.Sequential()
#cnn.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,1]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
#cnn.add(tf.keras.layers.Dense(units=50, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=17, activation='softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set,validation_data = testing_set,epochs = 25)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded / image'

try:
    import shutil
    shutil.rmtree('image')

except:
    pass

app = Flask(__name__)


@app.route('/')
def upload_f():
    return render_template('upload.html')

   

        


    pred = cnn.predict_generator(test_generator)
    print(pred)
    return eval(e)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        image = cv2.imread(f.filename)
        #image = cv2.resize(image,(300,300))
        #Converting the image to grayScale for better accuracy(RGB scale might underfit the data).
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #To smoothen the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # perform edge detection, find contours in the edge map, and sort the
        # resulting contours from left-to-right
        edged = cv2.Canny(blurred, 120, 50)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
        chars=[]
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # filter out bounding boxes, ensuring they are neither too small
            # nor too large
            if w*h>300 and (h>60 or w>40):
                # extract the character and threshold it to make the character
                # appear as *white* (foreground) on a *black* background, then
                # grab the width and height of the thresholded image
                #img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                #plt.imshow(img, cmap = 'gray')
                roi = gray[y-25:y + h+25, x-15:x + w+15]
                img = cv2.resize(roi,(64,64))
                norm_image = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                #norm_image=img/255
                norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
                case = np.asarray([norm_image])
                pred = cnn.predict([case])
                pred = np.argmax(pred, axis=1)
                
                chars.append(pred)
                cv2.rectangle(image, (x-15, y-25), (x + w + 15, y + h+25), (0, 255, 0), 2)
                
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        e = ''
        for i in chars:
            if i==0:
                e += '('
            elif i==1:
                e += ')'
            elif i==12:
                e += '+'
            elif i==13 or i==14:
                e += '/'
            elif i==15:
                e+='*'
            elif i==16:
                e+='-'
            else:
                e += str(i-2)
        try:
            v = eval(e)
        except:
            v = "can not be solved"
        return render_template('pred.html', ss = v)

if __name__ == '__main__':
    app.run()
