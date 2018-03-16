"""
This script builds and runs CNN models. 
It largely re-creates the work done in the notebook `CNN-Design.ipynb`.
It is here because I would like to train these models on a GPU instance on
Amazon's servers. 
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
import keras.backend as K
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import plot_model, to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import os
import argparse

# read command-line args, to see if the user wants to just run a sample
argparser = argparse.ArgumentParser()
argparser.add_argument("--arch", dest = "arch", default= 10, type = int,
				 help="Indicate which archetecture you want.")
argparser.add_argument("-s", "--sample", action="store_true",
                       help="Run only one epoch - for timing and testing purposes")
argparser.add_argument("-a", "--aws", action="store_true",
                       help="indicate if running aws (1) or not (0)")
args = argparser.parse_args()
# indicate which archetecture you want {0,1,2,3}
arch = args.arch
# indicate if only running a sample
sample = args.sample
# Indicate if running on aws or not
aws = args.aws

"""
Several CNN archetecture, of increasing complexity
"""
def model_3conv(height, width, depth, stride = 3, pool_size = 2):
    """
    CNN Keras model with 3 convolutions.
    
    parameters:
        - height: Shape of each image
        - stride: stride size
        
    return: 
        - A Keras `Sequential` model with an RMS-prop optimizer.
    """
    # set the input shape
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
    else:
        input_shape = (height,width,depth)
            
    model = Sequential()
    model.add(Conv2D(8, (stride, stride), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (stride, stride)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def model_4conv(height, width, depth, stride = 3, pool_size = 2):
    """
    CNN Keras model with 4 convolutions.
    
    parameters:
        - height,width,depth: Shape of each image
        - stride: stride size
        
    return: 
        - A Keras `Sequential` model with an RMS-prop optimizer.
    """
    # set the input shape
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
    else:
        input_shape = (height,width,depth)
            
    model = Sequential()
    model.add(Conv2D(8, (stride, stride), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (stride, stride)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def model_4conv2(height, width, depth, stride = 5, pool_size = 2):
    """
    Another CNN Keras model with 4 convolutions. 
    This one, however, emphasizes dimensionality reduction through larger stride length, 
    and fewer pooling layers
    
    parameters:
        - height,width,depth: Shape of each image
        - stride: stride size
        
    return: 
        - A Keras `Sequential` model with an RMS-prop optimizer.
    """
    # set the input shape
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
    else:
        input_shape = (height,width,depth)
            
    model = Sequential()
    model.add(Conv2D(8, (stride, stride), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (stride, stride)))
    model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
#     model.add(Dropout(0.25))
    
    model.add(Conv2D(16, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def model_6conv(height, width, depth, stride = 3, pool_size = 2):
    """
    CNN Keras model with 6 convolutions.
    
    parameters:
        - height,width,depth: Shape of each image
        - stride: stride size
        
    return: 
        - A Keras `Sequential` model with an RMS-prop optimizer.
    """
    # set the input shape
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
    else:
        input_shape = (height,width,depth)
            
    model = Sequential()
    model.add(Conv2D(8, (stride, stride), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (stride, stride)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (stride, stride), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

"""
Put all the archetecture generating functions in a list. 
Then, using the command line input, let the user decide which
archetecture when he/she runs the script. 
"""
model_factories = [model_3conv, model_4conv, model_4conv2, model_6conv]
# isolate the model generating function chosen by the user
model_factory = model_factories[arch]


"""
Load the dataset, and process it. 
This involves making train/test splits, and reshaping the images.

Relative paths of datasets will depend on if we're running on AWS or not. 
"""
if aws:
    X_path = "data/X_train.npy"
    y_path = "data/y_train.npy"
else:
    X_path = "../data/preproccessed/basic/X_train.npy"
    y_path = "../data/preproccessed/basic/y_train.npy"

# training set
with open(X_path, "rb") as handle:
    X = np.load(handle)
# Training labels
with open(y_path, "rb") as handle:
    y = np.load(handle)

"""
Create train/validation splits
"""
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1)

"""
Reshape output so that they are vectors, not matrices with one column. 

Also, reshape data, so that each image is 3d (has a depth of one - one channel)
The way this is done depends on the current Keras implementation, stored in the file
`~/.keraas/keras.json`. 
"""
# reshape the ouput vectors, so that they're vectors (not column matrices)
y_train = y_train.reshape((y_train.shape[0],))
y_valid = y_valid.reshape((y_valid.shape[0],))
# isolate training, validation and testing shapes, currently
trainshape, validshape = X_train.shape, X_valid.shape
# Reshape the X's, according to our channel setting. 
if K.image_data_format() == "channels_last":
    X_train = X_train.reshape((trainshape[0],trainshape[1],trainshape[2], 1))
    X_valid = X_valid.reshape((validshape[0],validshape[1],validshape[2], 1))
else:
    X_train = X_train.reshape((trainshape[0],1,trainshape[1],trainshape[2]))
    X_valid = X_valid.reshape((validshape[0],1,validshape[1],validshape[2]))

"""
Create a data augmentation scheme:
This will allow for random distortions to our data, which will increase
our models robustness to noise and hopefully reduce overfitting. 
"""
# construct the image generator for data augmentation
img_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)
# fit to our training data
img_gen.fit(X_train)


"""
Get the model the user specified for. 
This depends on the command line argument arch. 
The function `model_factory` will return the selected model. 
"""
# Get the model and optimizers
model, opt = model_factory(height = 28, width = 28, depth = 1)
# compile the model
model.compile(loss='categorical_crossentropy',
                      optimizer= opt,
                      metrics=['accuracy'])

"""
Currently labels are integers. 
They must be in one-hot encodings. 
Two quick functions for encoding/decoding the labels as one-hot vectors. 
"""
# train an encoder to convert labels to integer labels, one-hot encodings, and back
encoder = LabelEncoder().fit(y.reshape((y.shape[0],)))

def encode_onehot(lab):
    return to_categorical(encoder.transform(lab))

def decode_onehot(lab):
    return encoder.inverse_transform(np.argmax(lab, axis = 1))

"""
Prepare to train. 
Make sure that all the needed directories exist. 
Create needed checkpoints. 
"""
# a place to save trained models
if not os.path.isdir("models"):
    os.makedirs("models")
"""
Filepath to indicate where to save model
this will store the archetecture, and validation accuracy
in the name. 
"""
archname = ["_3_", "_4_", "_42_", "_6_"][arch]
filepath="models/weights_%sconv_{epoch:02d}-{val_loss:.2f}.hdf5" % (archname)

# Checkpointers for saving the model and early stopping
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
# create an Early Stopping callback
early_stopping = EarlyStopping(patience=8)

"""
Train!
"""
starttime = time.time()
# number of epochs is only one if the training flag was passed
# fit the model, and save the output
numepochs = (1 if sample else 100)
history = model.fit_generator(img_gen.flow(X_train, encode_onehot(y_train), batch_size=8),
        validation_data=(X_valid, encode_onehot(y_valid)), steps_per_epoch=len(X_train),
        epochs=numepochs, verbose=1, callbacks = [early_stopping, checkpointer])

print("Training time: {}".format(time.time() - starttime))

"""
Save the history for further investigation
"""
# where to store the training histories
if not os.path.isdir("history"):
    os.makedirs("history")
# name of history file
historyname = "history_%sconv_%s" % (archname,time.strftime("%h%d_%H%S"))
# save history
with open(historyname, "wb") as handle:
    pickle.dump(history.history, handle, protocol = 3)
