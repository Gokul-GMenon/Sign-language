"""
Model Architecture
"""

"""
Model Architecture
"""

import cv2
import numpy as np
import os
import tensorflow as tf

def predict(image, name):

    image = cv2.flip(image, 1) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sz = 128

    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    res = cv2.resize(res, (sz, sz))
    res = res/255.0
    image = np.reshape(res, (1, sz, sz))

    # Loading our own model
    model = tf.keras.models.load_model(os.path.join('Custom_models', name+'_Model.h5'))
    
        
    array = model.predict(image)


    x = np.argmax(array)
    letters = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z']
    return letters[x]
