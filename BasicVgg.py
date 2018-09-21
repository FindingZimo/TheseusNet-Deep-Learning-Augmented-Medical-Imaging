# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:10:27 2018

@author: Tara
"""


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, merge,Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dense, Dropout, Activation
from keras.layers import MaxPooling2D, UpSampling2D,Flatten,concatenate 
from keras import regularizers 
from keras.layers.normalization import BatchNormalization as bn
from keras import initializers
from keras import backend as K

#%%

def VGG19(input_specs, settings):
    
    input_shape = (
        input_specs["INPUT_Y_SIZE"],
        input_specs["INPUT_X_SIZE"],
        input_specs["n_channels"])
    
    # define the setting for the model
    KERNEL_SIZE = settings['KERNEL_SIZE']
    
    NUM_CLASSES = settings['NUM_CLASSES']
    drop_out_rate = settings['drop_out_rate']    
        
    # define input shape
    inputs = Input(shape = input_shape)
    
    # first conv block
    conv1 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    conv1 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
   # second conv block
    conv2 = Convolution2D(128, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(pool1)
    conv2 = Convolution2D(128, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # third conv block
    conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(pool2)
    conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv3)
    conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv3)
    conv3 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # fourth conv block
    conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(pool3)
    conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv4)
    conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv4)
    conv4 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # fifth conv block
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(pool4)
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv5)
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv5)
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    # six conv block
    conv6 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", kernel_initializer="he_normal")(pool5)
    
    # fully connected layer
    flat = Flatten(name = 'flatten')(conv6)
    fc = Dense(1024, activation="relu",name = 'fc1')(flat)  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    fc = Dropout( drop_out_rate )(fc)
    fc = Dense(1024, activation="relu", name = 'fc2')(fc)  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    fc = Dropout( drop_out_rate )(fc)
    #softmax activation for classifiers
    #fc = Dense(NUM_CLASSES, activation = "softmax", name = 'predictions')(fc)
    #linear activation for linear regression
    fc = Dense(NUM_CLASSES, activation = "linear", name = 'predictions', kernel_initializer="he_normal")(fc)

    # define model and compile
    model = Model(inputs=inputs, outputs=fc)
    return model