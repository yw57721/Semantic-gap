# -*- coding: utf-8 -*-
"""
modified on Richard's example

Created on Sat Mar 23 16:57:40 2019

@author: Li Xiang
"""
import sys
#sys.exit(0)
import math
import func_han
import func_eval
import numpy as np
import pandas as pd
#import cPickle
import _pickle as cPickle
from collections import defaultdict
import re
import random
random.seed(9001)

from bs4 import BeautifulSoup

import sys
import os


from nltk import tokenize

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers
from keras.models import load_model


MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
#VALIDATION_SPLIT = 0.2

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

#----------------1st step : let dataset be ready---------------------------------

#if dataset if not loaded into memory, we load it first
    
if(dir().count('data_train')==0):
    df=pd.read_csv("../data/original_dataset.csv",encoding = "ISO-8859-1")
    df=df[df.asin.str.startswith('B')]
    df=df.reset_index(drop=True)
    df = df.rename(columns={'reviews': 'review'})
    
    data_train=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
    data_val=pd.read_csv('..//data//generated_reviews.csv')
    data_val = data_val.rename(columns={'reviews': 'review'})
    
    print(data_train.shape)
    print(data_val.shape)
    
    train_reviews = []
    val_reviews = []
    all_texts = []
    
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx], "lxml")
        text = clean_str(text.get_text())
        all_texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        train_reviews.append(sentences)
    
    for idx in range(data_val.review.shape[0]):
        text = BeautifulSoup(data_val.review[idx], "lxml")
        text = clean_str(text.get_text())
        all_texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        val_reviews.append(sentences)
        
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(all_texts)

###sys.exit(0)

#-------------------set params for different purpose--------------------
cur_exp_param='cpu'#['cpu','ram','hd','gpu','screen']
print("cur_exp_param:",cur_exp_param,'\n')
#K=5 #top K results for evaluation
difference_percent=0.2 
#----------------------------end parm setting-------------------------

#--------------------get trainset-------------------------

asin_list=func_han.get_useful_asins(exp_param=cur_exp_param)

train_asin_map_review={}

for _asin in asin_list:
    index_list=data_train.index[data_train['asin'] == _asin].tolist()
    if(index_list!=[]):
        train_asin_map_review[_asin]=[train_reviews[i] for i in index_list]
    
asin_map_labels=func_han.get_sorted_label_list(exp_param=cur_exp_param)


train_review_label_dic={}
for _asin in train_asin_map_review:
    train_review_label_dic[_asin]=[train_asin_map_review[_asin],asin_map_labels[_asin]]
#-------------------get val set--------------------------

val_asin_map_review={}
for _asin in asin_list:
    index_list=data_val.index[data_val['asin'] == _asin].tolist()
    if(index_list!=[]):
        val_asin_map_review[_asin]=[val_reviews[i] for i in index_list]

val_review_label_dic={}
for _asin in val_asin_map_review:
    val_review_label_dic[_asin]=[val_asin_map_review[_asin],asin_map_labels[_asin]]

    
#-----------feed reviews into train/val set--------------------------------------
X_train_text=[]
y_train_all_labels=[]
#y_train=[]

X_val_text=[]
y_val_all_labels=[]
#y_val=[]

for asin in train_review_label_dic:
    len_cur_reviews=len(train_review_label_dic[asin][0])
    X_train_text.extend(train_review_label_dic[asin][0])
    for i in range(len_cur_reviews):
        y_train_all_labels.append(train_review_label_dic[asin][1])
#        y_train.append(train_review_label_dic[asin][1][0])
        
for asin in val_review_label_dic:
    len_cur_reviews=len(val_review_label_dic[asin][0])
    X_val_text.extend(val_review_label_dic[asin][0])
    for i in range(len_cur_reviews):
        y_val_all_labels.append(val_review_label_dic[asin][1])
#        y_val.append(val_review_label_dic[asin][1][0])

# -----------remember to shuffle the val set------------

val_concat=list(zip(X_val_text,y_val_all_labels))
random.shuffle(val_concat)

X_val_text,y_val_all_labels=zip(*val_concat)
X_val_text=list(X_val_text)


#------------get split point-----------
train_val_split=len(X_train_text)
val_test_split=int(len(X_val_text)/2)

train_val=X_train_text+X_val_text

#------get the word index and split into in train,val,test set

data = np.zeros((len(train_val), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(train_val):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

#--------------now set data ready--------------------------------------
                    
X_train=data[:train_val_split]
X_val=data[train_val_split:train_val_split+val_test_split]
X_test=data[train_val_split+val_test_split:]

#sys.exit(0)

y_train_all_labels
y_test_all_labels=y_val_all_labels[val_test_split:]
#y_test=y_val[val_test_split:]
y_val_all_labels=y_val_all_labels[:val_test_split]
#y_val=y_val[:val_test_split]

#change the random labels in y from 0 to N
#this is useful when we use to_categorical()

if(y_train_all_labels[0]!=[]):
    all_labels=sorted(list(set(y_train_all_labels[0])))
new_map_ylabel={}
for i in range(len(all_labels)):
    new_map_ylabel[all_labels[i]]=i

# new_map_ylabel is the dictionary we do this transformation
# now we get new label by map new_map_ylabel to each one

y_train_all_new=[]
y_val_all_new=[]
y_test_all_new=[]

for each_all_labels in y_train_all_labels:
    y_train_all_new.append(list(map(lambda x : new_map_ylabel[x],each_all_labels)))

for each_all_labels in y_val_all_labels:
    y_val_all_new.append(list(map(lambda x : new_map_ylabel[x],each_all_labels)))

for each_all_labels in y_test_all_labels:
    y_test_all_new.append(list(map(lambda x : new_map_ylabel[x],each_all_labels)))


y_train_all_labels=np.array(y_train_all_new)
y_train=np.array(y_train_all_labels)[:,0].tolist()

y_val_all_labels=np.array(y_val_all_new)
y_val=np.array(y_val_all_labels)[:,0].tolist()

y_test_all_labels=np.array(y_test_all_new)
y_test=np.array(y_test_all_labels)[:,0].tolist()


y_train_categorical = to_categorical(np.asarray(y_train))
y_val_categorical = to_categorical(np.asarray(y_val))
y_test_categorical = to_categorical(np.asarray(y_test))

output_classes=len(y_train_categorical[0])

#----------now train,val,test set are all ready--------------------


print('Shape of X_train:', X_train.shape)
print('Shape of X_val:', X_val.shape)
print('Shape of X_test:', X_test.shape)

print('\nShape of y_train_all_labels:', y_train_all_labels.shape)
print('Shape of y_val_all_labels:', y_val_all_labels.shape)
print('Shape of y_test_all_labels:', y_test_all_labels.shape)

#sys.exit(0)

#--------------prepare embedding layer----------------------------

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

GLOVE_DIR = "."
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
#        return mask
        return None
    

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
#apply sentEncoder to each of the MAX_SENTS timesteps
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)
preds = Dense(output_classes, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

sys.exit(0)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_acc', patience=2, verbose=2, mode='max')
model_name='./models/'+cur_exp_param+'_model.hdf5'
mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_acc', mode='max')

#print("model fitting - Hierachical attention network")
if(cur_exp_param=='cpu'):
    b_size=16
elif(cur_exp_param=='hd'):
    b_size=32
else:
    b_size=64
    
model.fit(X_train, y_train_categorical, validation_data=(X_val, y_val_categorical),
          epochs=5, batch_size=b_size, callbacks=[early_stopping,mcp_save])


#results=model.evaluate(X_test,y_test_categorical)

#----------------------calculate top-K PR value-------------------------

test_predict_prob=model.predict(X_test) 
test_predict=np.argmax(test_predict_prob, axis=1)

test_predict_top_5=np.argsort(-test_predict_prob, axis=1)[:,:5]
test_predict_top_10=np.argsort(-test_predict_prob, axis=1)[:,:10]
test_predict_all=np.argsort(-test_predict_prob, axis=1)

test_predict_top_k=test_predict_top_5
top_k=5

print('\nndcg:')
i = 0
ndcgs = []
if(math.floor(difference_percent*output_classes)==0):
    print("error: the relevant classes number is 0, \
          you may want to consider math.ceiling instead of math.floor\
          ")
labels_to_eval=y_test_all_labels[:,:math.floor(difference_percent*output_classes)]

while i < top_k:
    
    y_pred = test_predict_top_k[:, 0:i+1]
    i = i+1

    ndcg_i = func_eval._NDCG_score(y_pred,labels_to_eval)
    ndcgs.append(ndcg_i)

    print(ndcg_i)


#sys.exit(0)  

print("precision:")
i = 0
precisions = []

while i < top_k:
    
    y_pred = test_predict_top_k[:, 0:i+1]
    i = i+1

    precision = func_eval._precision_score(y_pred,labels_to_eval)
    precisions.append(precision)

    print(precision)

print("recall:")
i = 0
recalls = []
while i < top_k:
    
    y_pred = test_predict_top_k[:,  0:i+1]

    i = i+1   
    recall = func_eval.new_recall(y_pred, labels_to_eval)
    recalls.append(recall)

    print(recall)


#------------load best trained models--------------------------
#    if wanna load, run code to sys.exit(0) line, then run the following code
##    
##cur_exp_param='gpu'#['cpu','ram','hd','gpu','screen']
#filepath='./models/'+cur_exp_param+'_model.hdf5'
#
#
#if os.path.exists(filepath):
#    model.load_weights(filepath)