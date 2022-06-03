import cv2 as cv
import os

def train(name):

    # Path of the file containing the images
    PATH = os.path.join('training_images', name)
    
    # Importing the model and setting all the current layers as non-trainable

    from keras.models import model_from_json
    from keras import Sequential

    model = ''

    with open("model-bw-2-json.json") as json_file:
        model = model_from_json(json_file.read())
        model.load_weights("model-bw-2-weights.h5")

    final_model = Sequential()

    i=0
    for layer in model.layers[:4]:
        final_model.add(layer) 
        
    for layer in final_model.layers:
        layer.trainable = False

    import keras
    from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Convolution2D, MaxPooling2D

    final_model = keras.Sequential(final_model)

    # Second convolution layer and pooling
    final_model.add(Convolution2D(32, (3, 3), activation='relu'))
    # input_shape is going to be the pooled feature maps from the previous convolution layer
    final_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolution layer and pooling
    final_model.add(Convolution2D(32, (3, 3), activation='relu'))
    # input_shape is going to be the pooled feature maps from the previous convolution layer
    final_model.add(MaxPooling2D(pool_size=(2, 2)))


    # Flattening the layers
    final_model.add(Flatten())

    # Adding a fully connected layer
    final_model.add(Dense(units=128, activation='relu'))
    final_model.add(Dropout(0.40))
    final_model.add(Dense(96, activation='relu'))
    final_model.add(Dropout(0.40))
    final_model.add(Dense(64, activation='relu'))


    # Adding the final dense layer (only 1 as it is a binary class)
    final_model.add(Dense(27, activation='sigmoid'))

    import tensorflow as tf

    final_model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics=['accuracy']
    )

    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            zoom_range=0.5,
            width_shift_range=0.1,
            height_shift_range=0.1,)

    test_datagen = ImageDataGenerator(
            rescale=1./255)

    sz   = 128
    batch_size = 100

    data_dir_tr = os.path.join(PATH, 'train')
    data_dir_ts = os.path.join(PATH, 'test')


    training_set = train_datagen.flow_from_directory(data_dir_tr,
                                                    target_size=(sz, sz),
                                                    batch_size=18,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')

    test_set = test_datagen.flow_from_directory(data_dir_ts,
                                                target_size=(sz , sz),
                                                batch_size=10,
                                                color_mode='grayscale',
                                                class_mode='categorical') 

    final_model.fit(
                    training_set,
                    steps_per_epoch= 100, # No of images in training set
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=77)# No of images in test set
                    
    final_model.save(os.path.join('Custom_models', name + '_Model.h5'))