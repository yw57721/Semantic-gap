# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:55:20 2019

@author: Li Xiang

elmo embedding use tf-idf weight on the needs data


"""

import pandas as pd
import sys
import func_elmo
import h5py
import os 
import numpy as np
from func_tfidf import clean_review
from func_tfidf import get_inverted_index
from func_tfidf import get_max_tf
from func_tfidf import get_word_weight


#-------------------Predefined Functions--------------------

def load_test_review_emb(asin):
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
    
#-------------------------------------------------------------------------


asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

asin_map_needs={}
for _asin in asin_list:
    asin_map_needs[_asin]=load_test_review_emb(asin=_asin)

asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

#-------get the frequency of each word in a review ---------------------
#-------store in words_map_frequency (each key is a lowercaseword)------

df_needs=pd.read_csv("../data/generated_reviews.csv",encoding = "ISO-8859-1")
#df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
df_needs=df_needs[df_needs.reviews.apply(lambda x:len(str(x).split())<=500)]

asin_map_needs_words={}
total_needs_word_list=[]
for _asin in asin_list:
    asin_map_needs_words[_asin]=df_needs.reviews.loc[df_needs['asin']==_asin].tolist()

#----edit here----------
review_list=[]
for asin in asin_map_needs_words:
    review_list.extend(asin_map_needs_words[asin])
    
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
for asin in asin_map_needs_words:
#    print("{}".format(asin))
    length=len(asin_map_needs_words[asin])
    sent_embeds=np.zeros((length,1024))
    if(asin_map_needs[asin] is None):
        # if the elmo embedding doesn't exist
        # skip these reviews and 
        d_id+=len(asin_map_needs_words[asin])
        continue
    for i,sent in enumerate(clean_review(asin_map_needs_words[asin])):
        d_id+=1
        sent_embed=np.zeros((1024,))
        for j,w in enumerate(sent):
            
            sent_embed+=review_tfidf_dic['D'+str(d_id)][w]*\
                        asin_map_needs[asin][i][j]
            
        sent_embeds[i]=sent_embed
    asin_map_sentembedding[asin]=sent_embeds
    print("{} finished!".format(asin))

print("1 finished")


del asin_map_needs
#---------------train--------------------------------
  
#------------use asin_map_needs_sent_new and asin_map_labels to finish experiments--------------------
#validate: half of asin_map_needs_sent_new
#test: half of asin_map_needs_sent_new

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

embded_size=1024

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





























