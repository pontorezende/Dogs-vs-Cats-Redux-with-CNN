#!/usr/bin/env python
# coding: utf-8
from keras.preprocessing import image
from glob import glob
import cv2, os, random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#################################################################
################## PREPROCESSING THE DATA #######################
## all my files are inside 'train'
path = 'train/'

## used for resize and in our model
ROW, COL = 96, 96

dogs, cats = [], []
y_dogs, y_cats = [], []

## Definition to load all our dog images
def load_dogs():
    print('Loading all dog images\n')
    dog_path = os.path.join(path, 'dog*')
    for dog_img in glob(dog_path):
        dog = cv2.imread(dog_img)
        dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
        dog = cv2.resize(dog, (ROW, COL))
        dog = image.img_to_array(dog)
        dogs.append(dog)
    print('All dog images loaded')

## Definition to load all our cat images
def load_cats():
    print('Loading all cat images\n')
    cat_path = os.path.join(path, 'cat*')
    for cat_img in glob(cat_path):
        cat = cv2.imread(cat_img)
        cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
        cat = cv2.resize(cat, (ROW, COL))
        cat = image.img_to_array(cat)
        cats.append(cat)
    print('All cat images loaded')

print('#################################')
load_dogs()
print('#################################')
load_cats()

## classes for classification
classes = ['dog', 'cat']

## in case we want to see if our images was saved correctly in arrays we can use those codes
def show_dogs():
    plt.figure(figsize=(12,8))
    
    for i in range(5):
        plt.subplot(1, 5, i+1)
        img = image.array_to_img(random.choice(dogs))
        plt.imshow(img)
        
        plt.axis('off')
        plt.title('Supposed to be a {}'.format(classes[0]))
        
    plt.show()
def show_cats():
    plt.figure(figsize=(12,8))

    for i in range(5):
        plt.subplot(1, 5, i+1)
        img = image.array_to_img(random.choice(cats))
        plt.imshow(img)

        plt.axis('off')
        plt.title('Supposed to be a {}'.format(classes[1]))

    plt.show()

## just change the labels for 0 and  1
y_dogs = [1 for item in enumerate(dogs)]
y_cats = [0 for item in enumerate(cats)]

## converting everything to Numpy array to fit in our model
## them creating a X and target file like we used to see
## in Machine and Deep Learning models
dogs = np.asarray(dogs).astype('float32')
cats = np.asarray(cats).astype('float32')
y_dogs = np.asarray(y_dogs).astype('int32')
y_cats = np.asarray(y_cats).astype('int32')

## fit values between 0 and 1
dogs /= 255
cats /= 255

X = np.concatenate((dogs,cats), axis=0)
y = np.concatenate((y_dogs, y_cats), axis=0)

#################################################################
################ HERE BEGGINS OUR CNN MODEL #####################
print('#################################')
print('Setting up the CNN ...')
## in my case I need those lines to give some verbosity for TF with GPU
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
IMG_CHANNEL = 1
BATCH_SIZE = 128
N_EPOCH = 40
VERBOSE = 2
VALIDAION_SPLIT = .2
OPTIM = Adam()
N_CLASSES = len(classes)

## One-Hot Encoding
y = np_utils.to_categorical(y, N_CLASSES)
print('One-Hot Encoding done')

## Here is our model as a CNN
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(ROW, COL, IMG_CHANNEL), activation='relu'),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(.25),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(.5),
    Dense(N_CLASSES, activation='softmax')
])

print('The model was created by following config:')
model.summary()

## Compiling the model
print('Start compilation ...')
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
print('Compilation done')

## to save checkpoint to use latter
checkpoint = ModelCheckpoint('model_checkpoint/dogs_vs_cats_redux_checkpoint.h5')
print('#################################')
print('########### RUNNING #############')
model.fit(X, y, batch_size=BATCH_SIZE, epochs=N_EPOCH, validation_split=VALIDAION_SPLIT,
          verbose=VERBOSE, callbacks=[checkpoint])
print('Finished')

## scores
print('#################################')
print('############ SCORE ##############')
scores = model.evaluate(X, y, verbose=2)
print('MODEL ACCURACY\n{}: {}%'.format(model.metrics_names[1], scores[1]*100))
print('\n')
def saving_model():
    print('#################################')
    print('######### SAVING MODEL ##########')
    ## saving my model as
    model_name = 'Kaggle_Cats_Dogs_redux_CNN'
    print('\nSaving our model as {}'.format(model_name))

    ## saving architecture
    model_json = model.to_json()
    open(model_name+'.json', 'w').write(model_json)
    print('JSON saved')

    ## and the weights learned by our deep network on the training set
    model.save(model_name+'.h5', overwrite=True)
    print('.h5 saved')

    model.save_weights(model_name+'_weights.h5', overwrite=True)
    print('Weights saved in .h5 file')

saving_model()