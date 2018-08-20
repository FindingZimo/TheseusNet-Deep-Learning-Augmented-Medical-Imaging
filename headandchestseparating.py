# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:03:57 2018

@author: Zimo
"""
#import necessary things
#%load_ext autoreload
#%autoreload 2

from matplotlib import pyplot as plt
from IPython import display
from keras.optimizers import Adam, SGD
import numpy as np
from random import shuffle
import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
import h5py
import pandas as pd
# gets tensorflow backend to use in keras:
from keras import backend as K
import random
from scipy.stats import linregress


# file data from h5py
"""
Load image as number arrays 128x128x20 (slices)
and the clinical information (columns from the data sheet)
"""

FileName = "D:/Folder of things for Zimo/Img&Info.hdf5"
file = h5py.File(FileName, mode = "r")

#clinical data and tested variable
TestThis = "COPD"

test_arry = file[TestThis][()]

#load image arrays
img = file["input_img"][()]
file.close()


"""
choose to disclude any data
"""

    
X = img
X = X[:, :, :, :]
y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


"""
Specify parameters/settings
"""
input_specs = dict()
input_specs["INPUT_Y_SIZE"] = 256
input_specs["INPUT_X_SIZE"] = 256
input_specs["n_channels"] = 2

settings = dict()
#settings["KERNEL_SIZE"] = 7
settings['NUM_CLASSES'] = 2
settings['drop_out_rate'] = 0.35
EPOCHS = 100
batch_size = 4
NUM_VALIDATIONS = 1

model = VGG19(input_specs, settings)
DESTINATION = 'D:/Folder of things for Zimo/Runs/Week5/heads'

X_test = np.array((X[21], X[35], X[41], X[43], X[45], X[50], X[51], X[52], X[53], X[54]))
X_train = np.array((X[81], X[129], X[165], X[175], X[298], X[200], X[201], X[202], X[203], X[204]))
y_test = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_train = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_test = np_utils.to_categorical(y_test, settings['NUM_CLASSES'])
y_train = np_utils.to_categorical(y_train, settings['NUM_CLASSES'])

#%%
################ beginning of k-fold cross-validation routine ######################
#initialize the best loss as infinity
min_loss = float("inf")
best_val = 0

for ii in range(NUM_VALIDATIONS):
    
    #reset model at the start of every validation
    del model
    K.clear_session()
    print("\n\nVALIDATION: ", ii + 1, "/ ", NUM_VALIDATIONS)
    model = VGG19(input_specs, settings)
    model.compile(optimizer = Adam(lr = 1e-5, decay = 0.03),
                  loss = 'mean_squared_error', metrics = ['accuracy'])

    #begin training
    history = model.fit(X_train, y_train, 
          epochs=EPOCHS, 
          verbose=1,
          validation_data = (X_test ,y_test), 
          batch_size = batch_size,
          callbacks = [TensorBoard('C:/Users/Tara/tensorboard_output')])
    
    #save the validation that ends with the lowest loss function
    if min(history.history['val_loss']) < min_loss:
        best_val = ii
        min_loss = min(history.history['val_loss'])
        model.save(DESTINATION + '.h5')


#%%
y_pred = model.predict(X_test)

y_pred = y_pred[:, 1]

Nv = len(y_pred)

y_pred_col = y_pred.reshape(-1)
