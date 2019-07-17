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
from keras.optimizers import Adam
from keras.utils import to_categorical
#from keras import sparse_categorical_crossentropy

#Global Variables
folds=5

X=np.loadtxt('./clean_data/features.txt')
y=np.loadtxt('./clean_data/labels.txt')

X=np.reshape(X, (-1,1,26))

seed = 15
test_size = 0.1
X,y=shuffle(X,y, random_state=seed)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
kf = KFold(n_splits=folds,shuffle=False)

i=0
for train_index, test_index in kf.split(X):
    print('Round ',i)
    i+=1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train=to_categorical(y_train)
    
    model=models.build_1dCNN()
    
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=16, epochs= 2) #validation_data = (x_test, y_test)) #validation_data = (x_test, y_test))
    
    
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