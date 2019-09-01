# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:40:10 2019

@author: Li Xiang
"""
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import Input
from keras.models import Model



def rand_bilstm_sent_emb(x):
    
    n_timestep=499 #sentence length
    n_dimension=1024 #word embedding dimension
    out_shape=128 #output shape of each LSTM cell
    path='./checkpoint/random_bilstm.hdf5'
    
    weight_init=keras.initializers.RandomUniform(minval=-1/np.sqrt(n_timestep), maxval=1/np.sqrt(n_timestep), seed=None)
    inputs = Input(shape=(n_timestep,n_dimension))

    lstm_layer=LSTM(out_shape, return_sequences=True,kernel_initializer=weight_init,\
                recurrent_initializer=weight_init,\
                bias_initializer=weight_init,\
                )
    
    bilstm_layer=Bidirectional(lstm_layer, \
                               input_shape=(n_timestep,n_dimension),\
                               merge_mode='concat')
    
    predictions = bilstm_layer(inputs)
    
    bilstm_model = Model(inputs=inputs, outputs=predictions)
    
    bilstm_model.load_weights(path)
    
    #if test data sentence length smaller than 499
    new_x=np.zeros((x.shape[0],n_timestep,n_dimension))
#    print(x.shape)
    sentence_length=x.shape[1]#147
    padding=np.zeros((n_timestep-sentence_length,n_dimension))
    if(sentence_length<n_timestep):
        for i,each in enumerate(x):
            new_x[i]=np.concatenate((each, padding), axis=0)
            
#    print(new_x.shape)
    y=bilstm_model.predict(new_x)
    y=np.amax(y, axis=2)

    return y

