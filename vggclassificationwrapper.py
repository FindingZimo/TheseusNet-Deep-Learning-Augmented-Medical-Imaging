# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:03:57 2018

@author: Tara and Kang
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
from sklearn import metrics
from scipy.stats import linregress

# file data from h5py
"""
Load image as number arrays 128x128x20 (slices)
and the clinical information (columns from the data sheet)
"""

FileName = "D:/Folder of things for Zimo/img&info.hdf5"
#FileName = "D:/Folder of things for Zimo/Smalltestdataset9_7_18.hdf5"
#FileName = "D:/Folder of things for Zimo/fulldataset.hdf5"
file = h5py.File(FileName, mode = "r")

#clinical data and tested variable
TestThis = "COPD"

test_arry = file[TestThis][()]

#load image arrays
img = file["input_img"][()]
file.close()

#makes sure that NaNs are ignored
indx = ~np.isnan(test_arry)

for i in range(0, len(indx)):
    if test_arry[i] == 3:
        indx[i] = False

X = img[indx]
y = test_arry[indx]

"""
split data into 5 groups (folds)
"""
# splitting of the data
index = np.arange(len(y))
random.seed(a = 0)
random.shuffle(index)
# 5-fold validation
index_kfold = np.array_split(index,5)
y_pred = np.zeros((0, 2))
y_true = np.zeros((0, 2))

"""
Specify parameters/settings
"""
input_specs = dict()
input_specs["INPUT_Y_SIZE"] = 256
input_specs["INPUT_X_SIZE"] = 256
input_specs["n_channels"] = 20

settings = dict()
settings["KERNEL_SIZE"] = 5
settings['NUM_CLASSES'] = 2
settings['drop_out_rate'] = 0.2
EPOCHS = 50
batch_size = 8
NUM_VALIDATIONS = 1

model = VGG19(input_specs, settings)
DESTINATION = 'D:/Folder of things for Zimo/Runs/Week10/vggCOPD'

y = np_utils.to_categorical(y, settings['NUM_CLASSES'])

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
    model.compile(optimizer = Adam(lr = 1e-5, decay = 0.01),
                  loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #establish indices for each validation
    trainidx=np.array([],dtype=int)
    testidx = index_kfold[ii]
    tmp  = np.setdiff1d(np.arange(5),ii)
    for jj in tmp:  
        trainidx = np.concatenate((trainidx, index_kfold[jj])) 
    print("trainidx: " + str(trainidx))
    print("testidx: " + str(testidx)) 

    X_train = X[trainidx]
    X_test = X[testidx]
    y_train = y[trainidx]
    y_test = y[testidx]

    #begin training
    history = model.fit(X_train, y_train, 
          epochs=EPOCHS, 
          verbose=1,
          validation_data = (X_test, y_test), 
          batch_size = batch_size,
          callbacks = [TensorBoard('C:/Users/Tara/tensorboard_output')])
    
    #save the validation that ends with the lowest loss function
    if min(history.history['val_loss']) < min_loss:
        best_val = ii
        min_loss = min(history.history['val_loss'])
        model.save(DESTINATION + '.h5')

    tmp_y_pred = model.predict(X_test)
    y_pred = np.concatenate((y_pred,tmp_y_pred))
    
    tmp_y_true = y[testidx]
    y_true = np.concatenate((y_true, tmp_y_true))


#%%
Nv = len(y_pred)

y_pred_col = np.zeros(Nv)
y_true_col = np.zeros(Nv)

for i in range(0, Nv):
    y_pred_col[i] = y_pred[i][1]
    
for i in range(0, Nv):
    y_true_col[i] = y_true[i][1]

cross_val_idx = np.zeros((Nv,))
case_col = index[0:Nv]
ct = 0
for jj in range(NUM_VALIDATIONS):
    for kk in index_kfold[jj]:
        cross_val_idx[ct] = jj
        ct=ct+1

#%%
#create data_table to save data 

datatable = np.transpose(np.vstack((case_col,cross_val_idx,y_true_col,y_pred_col)))
df2 = pd.DataFrame({'case_index':case_col,'cross_validation_trial':cross_val_idx,'y_true':y_true_col,'y_pred':y_pred_col})

#make excel file
writeobj = pd.ExcelWriter(DESTINATION + '.xlsx')

df2.to_excel(writeobj, TestThis)
writeobj.save()

#make scatterplot
#plt.scatter(y_pred_col, y_true_col)
#plt.show()
#slp, intrcpt, r_val, p_val, std_err = linregress(y_pred_col, y_true_col)
#print("r-squared: {}".format(r_val * r_val))


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
