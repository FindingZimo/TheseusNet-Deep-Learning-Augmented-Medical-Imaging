# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:03:57 2018

@author: Tara and Kang
"""
#import necessary things

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
from scipy.stats import linregress


# file data from h5py
"""
This is the file with all the data
it has the images as number arrays 128x128x20 (slices)
and the clinical information (columns from the data sheet)

"""


FileName = "D:/Folder of things for Zimo/Img&Info.hdf5"
file = h5py.File(FileName, mode = "r")

#This is the clinicl data you want to test against
TestThis = "COPD"
test_arry = file[TestThis][()]

#This loads the image arrays
img = file["input_img"][()]
file.close()

#makes sure that blanks are ignored
indx = ~np.isnan(test_arry)

#make sure that 3s are ignored in the case of COPD variable
for i in range(0, len(indx)):
    if test_arry[i] == 3:
        indx[i] = False


test_arry = test_arry[indx]
img = img[indx]
X = img
y = test_arry



index = np.arange(len(y))
index_5fold = np.array_split(index,5)
y_pred = np.zeros((len(y), 2))




"""
Here you specify your parameters of the images
This is probably not going to change unless you start using different images
"""
input_specs = dict()
input_specs["INPUT_Y_SIZE"] = 256
input_specs["INPUT_X_SIZE"] = 256
input_specs["n_channels"] = 20


"""
Here you define your model settings.
Change Kernels, or Classes as needed.
The classes are how many different classes you are sorting the data into
Something that is binary would have 2 classes, etc
"""
# define the setting for the model
settings = dict()
settings["KERNEL_SIZE"] = 7
settings['NUM_CLASSES'] = 2
settings['drop_out_rate'] = 0.35
EPOCHS = 100
batch_size = 5



"""
This is the code that runs it all
Learning rate (lr) and decay can be changed
Optimizers can be changed
You can ask for whatever metrics you want, such as accuracy

cd tensorboard_output
tensorboard --logdir=.
"""



#%%
################ beginning of fitting ######################
model = VGG19(input_specs, settings)
model.compile(optimizer = Adam(lr = 1e-5, decay = 0.03),
              loss = 'binary_crossentropy', metrics = ['accuracy'])


X_train = X[58:len(y)]#np.concatenate((X[0:11], X[21:291]))
X_test = X[0:58]

y_train = y[58:len(y)]#np.concatenate((y[0:11], y[21:291]))
y_test = y[0:58]

y_train = np_utils.to_categorical(y_train, settings['NUM_CLASSES'])
y_test = np_utils.to_categorical(y_test, settings['NUM_CLASSES'])

model.fit(X_train, y_train, 
          epochs=EPOCHS, 
          verbose=1,
          validation_data = (X_test, y_test), 
          batch_size = batch_size,
          callbacks = [TensorBoard('C:/Users/Tara/tensorboard_output')])

y_pred = model.predict(X)
    
    
#%%
#You need to change the name of the model when you save it
model.save('D:/Folder of things for Zimo/Runs/week3/classifypackyears.h5')

#%% START THIS AFTER THE MODEL RUNS

#this makes a scatter plot of data
#not necessary for classifications
#plt.scatter(y_pred, y[index])



#%%

Nv = len(y)
y_pred_col = np.zeros((Nv,))

for ii in range(Nv):
    y_pred_col[ii] = y_pred[ii][1]

y_true_col = y[index]
case_col = index
cross_val_idx = np.zeros((Nv,))
ct = 0

for jj in range(5):
    for kk in index_5fold[jj]:
        cross_val_idx[ct] = jj
        ct = ct+1


tested = np.zeros(len(y))

for i in tested:
    i = False
for j in range(0, 58):
    tested[j] = True

#%%

# create data_table to save data 
    
datatable = np.transpose(np.vstack((case_col,cross_val_idx,y_true_col,y_pred_col)))
df2 = pd.DataFrame({'case_index':case_col, 'tested':tested, 'y_true':y_true_col, 'y_pred':y_pred_col})


#%%
#This makes an excel file out of the data
writeobj = pd.ExcelWriter('D:/Folder of things for Zimo/Runs/week3/classifypackyears.xlsx') 

#This should be whatever variable you are testing
df2.to_excel(writeobj, TestThis)
writeobj.save()


#graph only the test data
y_pred_col = model.predict(X_test)
y_pred_col = y_pred_col[:, 1]
y_true_col = y_test[:, 1]


#print out true and false positives and negatives
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

for i in range(0, len(y_pred_col)):
    if((y_pred_col[i] > 0.5) & (y_true_col[i] == 1)):
        true_positive += 1
    elif((y_pred_col[i] <= 0.5) & (y_true_col[i] == 0)):
        true_negative += 1
    elif((y_pred_col[i] > 0.5) & (y_true_col[i] == 0)):
        false_positive += 1
    elif((y_pred_col[i] <= 0.5) & (y_true_col[i] == 1)):
        false_negative += 1
print("(TP, TN, FP, FN) = ({}, {}, {}, {})".format(true_positive, true_negative, false_positive, false_negative))


#plot overall ROC curve
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_true_col, y_pred_col)
metrics.roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% metrics.roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.show()

slp, intrcpt, r_val, p_val, std_err = linregress(y_pred_col, y_true_col)
print("r-coefficient: ", r_val)


#plot individual ROC curve
#y_pred0 = y_pred_col[0:59]
#y_pred1 = y_pred_col[59:118]
#y_pred2 = y_pred_col[118:175]
#y_pred3 = y_pred_col[175:233]
#y_pred4 = y_pred_col[233:291]
#
#y_true0 = y_true_col[0:59]
#y_true1 = y_true_col[59:118]
#y_true2 = y_true_col[118:175]
#y_true3 = y_true_col[175:233]
#y_true4 = y_true_col[233:291]
#
#fpr0, tpr0, thresholds = metrics.roc_curve(y_true0, y_pred0)
#fpr1, tpr1, thresholds = metrics.roc_curve(y_true1, y_pred1)
#fpr2, tpr2, thresholds = metrics.roc_curve(y_true2, y_pred2)
#fpr3, tpr3, thresholds = metrics.roc_curve(y_true3, y_pred3)
#fpr4, tpr4, thresholds = metrics.roc_curve(y_true4, y_pred4)
#
#metrics.roc_auc0 = metrics.auc(fpr0, tpr0)
#metrics.roc_auc1 = metrics.auc(fpr1, tpr1)
#metrics.roc_auc2 = metrics.auc(fpr2, tpr2)
#metrics.roc_auc3 = metrics.auc(fpr3, tpr3)
#metrics.roc_auc4 = metrics.auc(fpr4, tpr4)
#
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr0, tpr0, 'red',
#label='Validation 0 AUC = %0.2f'% metrics.roc_auc0)
#plt.plot(fpr1, tpr1, 'orange',
#label='Validation 1 AUC = %0.2f'% metrics.roc_auc1)
#plt.plot(fpr2, tpr2, 'yellow',
#label='Validation 2 AUC = %0.2f'% metrics.roc_auc2)
#plt.plot(fpr3, tpr3, 'green',
#label='Validation 3 AUC = %0.2f'% metrics.roc_auc3)
#plt.plot(fpr4, tpr4, 'blue',
#label='Validation 4 AUC = %0.2f'% metrics.roc_auc4)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.1,1.2])
#plt.ylim([-0.1,1.2])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()



