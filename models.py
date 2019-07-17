# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:57:43 2019

@author: Dacong
"""

from keras.models import Sequential
from keras.layers import MaxPooling1D,Conv1D,Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, merge, Input, Dense, Dropout, SimpleRNN, LSTM,TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import to_categorical

def build_1dCNN():
    model = Sequential()
    model.add(Conv1D(filters=50,
                     kernel_size=3,
                     strides=1,
                     input_shape=(1, 26),
                     padding='same'      # Padding method
                     ))   
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='same'    # Padding method
                           ))
    model.add(Conv1D(filters=100,
                     kernel_size=3,
                     strides=1,
                     padding='same'      # Padding method
                     ))   
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='same'    # Padding method
                           ))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model