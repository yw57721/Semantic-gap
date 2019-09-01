# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:19:13 2019

@author: Li Xiang

elmo embedding use tf-idf weight on the amazon review data

"""

import sys
import func_elmo
import h5py
import os 
import numpy as np
import pandas as pd
import re
from math import log

from func_tfidf import clean_review
from func_tfidf import get_inverted_index
from func_tfidf import get_max_tf
from func_tfidf import get_word_weight

"""
#-------------------Predefined Functions--------------------
"""

def load_train_review_emb(asin):
    """
    return current embeddings in a numpy array
    based on asin number
    
    """
    filepath='D:\\Datasets\\elmo\\elmo_train_reviewdata\\'+asin+'.h5'
    if(os.path.isfile(filepath)):
        with h5py.File(filepath,'r') as hf:
            cur_sent_embd=hf[asin][:]
        return cur_sent_embd 
    else:
        return None


#-------------------set params for different purpose--------------------

warm_start_set=True#True#False
cur_exp_param='ram'#['cpu','ram','hd','gpu','screen']
#cur_sent_embd_type='max'#['max','ave','concat']
classifier_type='mlp'#'svm''mlp'
K=5 #top K results for evaluation
difference_percent=0.2 

#----------------------------end parm setting-------------------------

asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

asin_map_reviews={}
for _asin in asin_list:
    asin_map_reviews[_asin]=load_train_review_emb(asin=_asin)

asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

asin_map_reviews['B079QZ5DD7'].shape

#-------get the frequency of each word in a review ---------------------
#-------store in words_map_frequency (each key is a lowercaseword)------

df=pd.read_csv("../data/original_dataset.csv",encoding = "ISO-8859-1")
df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
df=df[df.reviews.apply(lambda x:len(str(x).split())<=500)]

df=df[df.asin.str.startswith('B')]
#sys.exit(0)
asin_map_review_words={}
total_word_list=[]
for _asin in asin_list:
    asin_map_review_words[_asin]=df.reviews.loc[df['asin']==_asin].tolist()

#----edit here----------
review_list=[]
for asin in asin_map_review_words:
    review_list.extend(asin_map_review_words[asin])
    
review_list = clean_review(review_list)    
inverted_index = get_inverted_index(review_list)
print("inverted_index finished!")

#---------------------tf-idf--------------------------------------
#------calculate max_tf-----------
max_tf = get_max_tf(review_list)
#-------get each document vector(save all in doc_vec_dic) ---------
review_dic={}
for d in range(len(review_list)):
    did='D'+str(d+1)
    review_dic[did]=review_list[d]

review_tfidf_dic={}
for i in range(len(review_list)):
    dID='D'+str(i+1)
    review_tfidf_dic[dID]=get_word_weight(dID,inverted_index,max_tf,review_dic[dID])

print("tf_idf dic finished!")

"""
------------compute sentence embeddings
#----------construct sentence embeddings based on ----------------------
#----------asin_map_review_words and asin_map_labels ---------------------

"""
#----------1.compute sent embeddings use tf-idf weight----------------------

asin_map_sentembedding={}
d_id=0
for asin in asin_map_review_words:
#    print("{}".format(asin))
    length=len(asin_map_review_words[asin])
    sent_embeds=np.zeros((length,1024))
    if(asin_map_reviews[asin] is None):
        # if the elmo embedding doesn't exist
        # skip these reviews and 
        d_id+=len(asin_map_review_words[asin])
        continue
    for i,sent in enumerate(clean_review(asin_map_review_words[asin])):
        d_id+=1
        sent_embed=np.zeros((1024,))
        for j,w in enumerate(sent):
            
            sent_embed+=review_tfidf_dic['D'+str(d_id)][w]*\
                        asin_map_reviews[asin][i][j]
            
        sent_embeds[i]=sent_embed
    asin_map_sentembedding[asin]=sent_embeds
    print("{} finished!".format(asin))

print("1 finished")

del asin_map_reviews

#---------------train--------------------------------
    
train_list=[]
for asin in asin_map_sentembedding:
    if(len(asin_map_sentembedding[asin])==1):
        train_list.append((asin_map_sentembedding[asin],asin_map_labels[asin]))
    else:
        for each in asin_map_sentembedding[asin]:
            train_list.append((each,asin_map_labels[asin]))

embded_size=1024

X_train=np.zeros((len(train_list),embded_size))
y_train_all_labels=[]
y_train=[]

for i,s in enumerate(train_list):
    X_train[i]=s[0]
    y_train_all_labels.append(s[1])
    y_train.append(s[1][0])
    
    
    
    
    
    