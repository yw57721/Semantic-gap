# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:40:41 2019

@author: Li Xiang

get random bilstm sentence embedding on the needs data

"""



import sys
import func_elmo
import h5py
import os 
import numpy as np
import pandas as pd
import re
from math import log

from func_rand_bilstm import rand_bilstm_sent_emb

"""
#-------------------Predefined Functions--------------------
"""

def load_train_review_emb(asin):
    """
    return current embeddings in a numpy array
    based on asin number
    
    """
    filepath='D:\\Datasets\\elmo\\elmo_generate_reviewdata\\'+asin+'.h5'
    if(os.path.isfile(filepath)):
        with h5py.File(filepath,'r') as hf:
            cur_sent_embd=hf[asin][:]
        return cur_sent_embd 
    else:
        return None


#-------------------set params for different purpose--------------------

warm_start_set=True#True#False
cur_exp_param='cpu'#['cpu','ram','hd','gpu','screen']
#cur_sent_embd_type='max'#['max','ave','concat']
classifier_type='mlp'#'svm''mlp'
K=5 #top K results for evaluation
difference_percent=0.2 

#----------------------------end parm setting-------------------------

asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

asin_map_reviews={}
for _asin in asin_list:
    asin_map_reviews[_asin]=load_train_review_emb(asin=_asin)

print("load elmo embedding finished.\n now generate random_bilstm sent embedding....")
asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

asin_map_sentembedding={}

length=len(asin_map_reviews.keys())
for i,asin in enumerate(asin_map_reviews):
    
    elmo_emb=asin_map_reviews[asin]
    if(elmo_emb is not None):
        asin_map_sentembedding[asin]=rand_bilstm_sent_emb(elmo_emb)
    print("{}/{} finished".format(i+1,length))

filepath='.\\checkpoint\\'+cur_exp_param+'_test.h5'

if(not(os.path.isfile(filepath))):    
    with h5py.File(filepath, 'w-') as hf:
        for asin in asin_map_sentembedding:
            hf.create_dataset(asin,  data=asin_map_sentembedding[asin])

new_sent_embd={}
if(os.path.isfile(filepath)):
    with h5py.File(filepath,'r') as hf:
        for asin in asin_list:
            if(asin in hf):
                new_sent_embd[asin]=hf[asin][:]

asin_map_sentembedding=new_sent_embd

#-----------------------store to disk----------------------------------

filepath='.\\checkpoint\\'+cur_exp_param+'_test.h5'

if(not(os.path.isfile(filepath))):    
    with h5py.File(filepath, 'w-') as hf:
        for asin in asin_map_sentembedding:
            hf.create_dataset(asin,  data=asin_map_sentembedding[asin])

new_sent_embd={}
if(os.path.isfile(filepath)):
    with h5py.File(filepath,'r') as hf:
        for asin in asin_list:
            if(asin in hf):
                new_sent_embd[asin]=hf[asin][:]

asin_map_sentembedding=new_sent_embd

#----------------------prepare val and test--------------------------------

val_test_list=[]

for asin in asin_map_sentembedding:
    if(len(asin_map_sentembedding[asin])==1):
        val_test_list.append((asin_map_sentembedding[asin],asin_map_labels[asin]))
    else:
        for each in asin_map_sentembedding[asin]:
            val_test_list.append((each,asin_map_labels[asin]))

import random
random.seed(230)
random.shuffle(val_test_list)

embded_size=499

X_=np.zeros((len(val_test_list),embded_size))
y_all_labels=[]
y_=[]

for i,s in enumerate(val_test_list):
    X_[i]=s[0]
    y_all_labels.append(s[1])
    y_.append(s[1][0])

split=int(len(val_test_list)/2)

X_val=X_[:split]
X_test=X_[split:]

y_val=y_[:split]
y_val_all_labels=y_all_labels[:split]
y_test=y_[split:]
y_test_all_labels=y_all_labels[split:]

y_test_all_labels=np.array(y_test_all_labels)

