# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:40:31 2019

@author: Li Xiang

"""
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

n_timestep=499 #sentence length, hidden size
n_dimension=1024 #word embedding dimension
out_shape=128 # output shape of each lstm cell

weight_init=keras.initializers.RandomUniform(minval=-1/np.sqrt(n_timestep), maxval=1/np.sqrt(n_timestep), seed=None)
inputs = Input(shape=(n_timestep,n_dimension))

lstm_layer=LSTM(out_shape, return_sequences=True,kernel_initializer=weight_init,\
            recurrent_initializer=weight_init,\
            bias_initializer=weight_init,\
            )

bilstm_layer=Bidirectional(lstm_layer, input_shape=(n_timestep,\
n_dimension),merge_mode='concat')

predictions = bilstm_layer(inputs)

bilstm_model = Model(inputs=inputs, outputs=predictions)

# uncomment if want to generate new model
#bilstm_model.save_weights('./checkpoint/random_bilstm.hdf5')
