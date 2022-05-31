"""
Model Architecture
"""

"""
Model Architecture
"""

import cv2
import numpy as np
import tensorflow as tf

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sz = 128

"""
Model Architecture
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
#classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=27, activation='softmax')) # softmax for more than 2


def predict(image, name):
    # from keras import models

    image = cv2.flip(image, 1) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sz = 128

    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    res = cv2.resize(res, (sz, sz))
    # import cv2 as cv
    # cv.imshow('crop', res)
    # cv.waitKey(1)

    
    # image = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    
    # print('\nshape after gray - ',np.shape(image), end='\n\n')
    res = res/255.0
    image = np.reshape(res, (1, sz, sz))
    # image = np.reshape(res, (1, sz, sz,3))
    

    # model = classifier.load_weights('model-bw-2-weights.h5')
    # model = classifier.load_weights('Models-final\gokul_Model.h5.h5')
    # model = tf.keras.models.load_model('model-bw-2-weights.h5')
    # # image = np.array([image])
    # image = image/255

    # # Loading the model
    # with open('model-bw-2-json.json', 'r') as json_file:
    #     json_savedModel= json_file.read()

    # #load the model architecture 
    # model = tf.keras.models.model_from_json(json_savedModel)
    # model.load_weights('model-bw-2-weights.h5')

    # # Loading our own model
    model = tf.keras.models.load_model(os.path.join('Custom_models', name+'_Model.h5'))
    

    # Downloaded model
    # model = classifier.load_model('Downloaded model\\asl_classifier.h5')
        
    array = model.predict(image)


    x = np.argmax(array)
    # print('preidicted index - ',x, end =' ')
    # letters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z']
    letters = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z']



    # print('length - ',len(array[0]))
    
    return letters[x]
