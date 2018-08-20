from matplotlib import pyplot as plt
from IPython import display
from keras.optimizers import Adam, SGD
import numpy as np
from random import shuffle
import keras.utils
from keras.preprocessing.image import ImageDataGenerator
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
import h5py
import pandas as pd
# gets tensorflow backend to use in keras:
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
import random
from sklearn import metrics


FileName = "D:/Folder of things for Zimo/Img&Info.hdf5"
file = h5py.File(FileName, mode = "r")

#This is the clinicl data you want to test against
#This TestThis variable should be changed as needed
TestThis = "COPD"
test_arry = file[TestThis][()]

#This loads the image arrays
img = file["input_img"][()]
file.close()
        
g = np.zeros((256, 256))

X = img
y = test_arry


g = np.zeros((256, 256))


#for scan in range(19, -1, -1):
for scan in range(0, 20):
    g = X[298, :, :, scan]
    plt.imshow(g)
    plt.show()
    print("#", scan)

