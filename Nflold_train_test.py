# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:41:32 2019

@author: Dacong
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import tensorflow as tf
import models
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import to_categorical
#from keras import sparse_categorical_crossentropy

#Global Variables
folds=3

X=np.loadtxt('./clean_data/features.txt')
y=np.loadtxt('./clean_data/labels.txt')

#
#X=X[0:58080]
#X=np.reshape(X, (-1,4,26))
#
#y=y[0::4]

#convert discrete samples to time sequences
time_steps=0 #take into consider #time_steps frame on left and those on right
res = []
for index in range(time_steps, X.shape[0]-time_steps):
    res.append(X[index-time_steps: index + time_steps+1, :])
res = np.array(res)
X=res
y=y[time_steps: len(y)-time_steps]


seed = 15
#test_size = 0.1
#X,y=shuffle(X,y, random_state=seed)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
kf = KFold(n_splits=folds,shuffle=False)

#model training
i=0
for train_index, test_index in kf.split(X):
    print('Round ',i)
    i+=1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train=to_categorical(y_train)
    
    model=models.build_1dCNN(shape=(2*time_steps+1,26))
    model.summary()
    
    tensorboard = TensorBoard(log_dir='./tensorboard_output', histogram_freq=0, write_graph=True, write_images=False)
     
    model.fit(X_train, y_train, batch_size=100, epochs= 1, callbacks=[tensorboard]) #validation_data = (x_test, y_test)) #validation_data = (x_test, y_test))
    
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    accuracy=accuracy_score(y_test, y_pred)
    F_measure1=f1_score(y_test, y_pred, average='weighted')
    F_measure2=f1_score(y_test, y_pred, average='macro')
    F_measure3=f1_score(y_test, y_pred, average='micro')
    confusion_mat=confusion_matrix(y_test, y_pred)
    
    print('f1 score:weighted\t',F_measure1)
    print('f1 score:macro\t',F_measure2)
    print('f1 score:micro\t',F_measure3)
    print('accuracy:\t', accuracy)
    print('confusion matrix:\n', confusion_mat)
    

print("Final Measure on Validation Set:")    
X_valid=np.loadtxt('./clean_data/features_valid.txt')
y_valid=np.loadtxt('./clean_data/labels_valid.txt')
res = []
for index in range(time_steps, X_valid.shape[0]-time_steps):
    res.append(X_valid[index-time_steps: index + time_steps+1, :])
res = np.array(res)
X_valid=res
y_valid=y_valid[time_steps: len(y_valid)-time_steps ]

y_pred = np.argmax(model.predict(X_valid), axis=1)
accuracy=accuracy_score(y_valid, y_pred)
F_measure1=f1_score(y_valid, y_pred, average='weighted')
F_measure2=f1_score(y_valid, y_pred, average='macro')
F_measure3=f1_score(y_valid, y_pred, average='micro')
confusion_mat=confusion_matrix(y_valid, y_pred)

print('f1 score:weighted\t',F_measure1)
print('f1 score:macro\t',F_measure2)
print('f1 score:micro\t',F_measure3)
print('accuracy:\t', accuracy)
print('confusion matrix:\n', confusion_mat)