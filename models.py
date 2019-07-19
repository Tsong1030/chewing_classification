# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:57:43 2019

@author: Dacong
"""

from keras.models import Sequential
from keras.layers import Reshape, MaxPooling1D,Conv1D,Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, merge, Input, Dense, Dropout, SimpleRNN, LSTM,TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import to_categorical

def build_1dCNN( shape ):
    model = Sequential()
    model.add(Conv1D(filters=50,
                     kernel_size=3,
                     strides=1,
                     input_shape=shape,
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
    
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_1dCNN_LSTM():
    model = Sequential()
    model.add(Reshape((1, 26), input_shape=(26,)))
    model.add(Conv1D(filters=50,
                     kernel_size=3,
                     strides=1,
                     input_shape=(1, 26),
                     padding='same'      # Padding method
                     ))   
#    model.add(MaxPooling1D(pool_size=2,
#                           strides=2,
#                           padding='same'    # Padding method
#                           ))
    model.add(LSTM(units=50))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_RNN(shape):
    model = Sequential()
    model.add(LSTM(
                units=50,
                input_shape=shape,
                return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
                units=100,
                return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
    return model

def build_simpleRNN(shape):
    model = Sequential()
    model.add(SimpleRNN(
                units=50,
                input_shape=shape,
                return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
    return model
