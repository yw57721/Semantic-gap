# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:48:44 2019

@author: Li Xiang

train and finetune the MLP classifier and save to folder (./models/)

*run this file will set the dataset ready

"""

import pandas as pd
import re

import sys
#sys.exit(0)
import math
import func_glove
import func_eval
import numpy as np
import pandas as pd
#import cPickle
import _pickle as cPickle
from collections import defaultdict
import random
random.seed(9001)

from bs4 import BeautifulSoup
import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

from sklearn.neural_network import MLPClassifier
from keras.layers import Embedding
from sklearn.externals import joblib
from nltk import tokenize

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
#--------------set params-----------------------------------------------------

cur_exp_param='cpu'#['cpu','ram','hd','gpu','screen']
#K=5 #top K results for evaluation
difference_percent=0.2 
cur_sent_embd_type='ave'#['ave','max','concat','hier']
K=5 #top K results for evaluation

print("cur_exp_param:",cur_exp_param)
print("cur_sent_embd_type:",cur_sent_embd_type,'\n')
#--------------end setting params-----------------------------------------------


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
        text = BeautifulSoup(data_train.review[idx], "lxml").get_text()
#        text = clean_str(text)
        all_texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        train_reviews.append(sentences)
    
    for idx in range(data_val.review.shape[0]):
        text = BeautifulSoup(data_val.review[idx], "lxml").get_text()
#        text = clean_str(text)
        all_texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        val_reviews.append(sentences)
        
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(all_texts)
    
#--------------------get trainset-------------------------

asin_list=func_glove.get_useful_asins(exp_param=cur_exp_param)

train_asin_map_review={}

for _asin in asin_list:
    index_list=data_train.index[data_train['asin'] == _asin].tolist()
    if(index_list!=[]):
        train_asin_map_review[_asin]=[train_reviews[i] for i in index_list]
    
asin_map_labels=func_glove.get_sorted_label_list(exp_param=cur_exp_param)


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
#---------X_train,X_val,X_test stores indexes of words
print('Shape of X_train:', X_train.shape)
print('Shape of X_val:', X_val.shape)
print('Shape of X_test:', X_test.shape)

print('\nShape of y_train_all_labels:', y_train_all_labels.shape)
print('Shape of y_val_all_labels:', y_val_all_labels.shape)
print('Shape of y_test_all_labels:', y_test_all_labels.shape)

#sys.exit(0)
output_classes=len(y_train_categorical[0])

#--------------prepare embedding matrix----------------------------

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

if(dir().count('embedding_matrix')==0):
    GLOVE_DIR = "."
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
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

#----------Transformer X_train,X_val,X_test to ------------------------------
#----------X_train_emb, X_val_emb, X_test_emb ------------------------------
#----------they are computed from 4 sent embeding methods --------------------------------
     
X_train_emb=func_glove.get_sent_embed(X_train,cur_sent_embd_type,EMBEDDING_DIM,embedding_matrix)
X_val_emb=func_glove.get_sent_embed(X_val,cur_sent_embd_type,EMBEDDING_DIM,embedding_matrix)
X_test_emb=func_glove.get_sent_embed(X_test,cur_sent_embd_type,EMBEDDING_DIM,embedding_matrix)

"""
#if want to train and save classifier, comment this line
#if want to get train/test set ready, uncomment this line
"""
sys.exit(0)

#-----now use MLP to train (X_train_emb,y_train_categorical)---------------
#----------and fintune by (X_val_emb,y_val_categorical)-------------------

#---------use sklearn MLPClassifier----------------------------------------
#if(cur_sent_embd_type!='concat'):
#    hidden_neurons=256
#else:
hidden_neurons=256
    
classifier = MLPClassifier(hidden_layer_sizes=(hidden_neurons,hidden_neurons), max_iter=1000, alpha=0.001,
                     solver='adam', verbose=2,  random_state=21)

print("start training..")
classifier.fit(X_train_emb,y_train)

new_coefs=classifier.coefs_[:2]

if(cur_exp_param=='cpu'):
#        new_coefs.append(np.random.rand(100, 23))
    new_coefs.append(np.zeros((100, 23)))
if(cur_exp_param=='ram'):
    new_coefs.append(np.random.rand(100, 6))
if(cur_exp_param=='hd'):
    new_coefs.append(np.random.rand(100, 11))
if(cur_exp_param=='gpu'):
    new_coefs.append(np.random.rand(100, 8))
if(cur_exp_param=='screen'):
    new_coefs.append(np.random.rand(100, 9))

finetune_classifier=MLPClassifier(hidden_layer_sizes=(hidden_neurons,hidden_neurons), max_iter=1000, alpha=0.001,
                     solver='adam', verbose=1,  random_state=21)

finetune_classifier.coefs_=new_coefs
finetune_classifier.fit(X_val_emb,y_val_categorical)

#-----give prediction on X_test_emb by review length and label frequency--------------------

model_name='./models/'+cur_exp_param+'_'+cur_sent_embd_type+'_model.pkl'
joblib.dump(finetune_classifier, model_name)


print("now classifier is finutuned and ready to test")
print("use finetune_classifier and X_test_emb")


#sys.exit(0)

