# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:44:58 2019

@author: Li Xiang

this file apply MR to test set

"""
import pandas as pd
import sys
import func_elmo
import h5py
import os 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

"""
#-------------------Predefined Functions--------------------
"""
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
    total_needs_word_list.extend(asin_map_needs_words[_asin])
    
a=list(map(lambda x:x.split(),total_needs_word_list))
#all words to lowercase
total_needs_word_list=[item.lower() for sublist in a for item in sublist]
#total_word_list = [re.sub(r"[^A-Za-z]+", '', item).lower() for sublist in a for item in sublist]
total_needs_word_len=len(total_needs_word_list)
#sys.exit(0)
needs_words_set=set(total_needs_word_list)

needs_words_map_frequency={}
for w in needs_words_set:
    needs_words_map_frequency[w]=total_needs_word_list.count(w)/total_needs_word_len
"""
------------compute sentence embeddings
#----------construct sentence embeddings based on ----------------------
#----------asin_map_review_words and asin_map_labels ---------------------

"""
#------------------1.compute sent embeddings---------------------------

a=10**-4

asin_map_needs_sentembedding={}

for asin in asin_map_needs_words:
#    print("{}".format(asin))
    length=len(asin_map_needs_words[asin])
    sent_embeds=np.zeros((length,1024))
    if(asin_map_needs[asin] is None):
        continue
    for i,sent in enumerate(asin_map_needs_words[asin]):
        sent_embed=np.zeros((1024,))
        for j,w in enumerate(sent.split()):
            
            sent_embed+=a/(a+needs_words_map_frequency[w.lower()])* \
                        asin_map_needs[asin][i][j]
        sent_embeds[i]=sent_embed
    asin_map_needs_sentembedding[asin]=sent_embeds

print("1 finished")

#------------------2.form the matrix X_sentence(cols are sent embeddings)---------------------------

num_sents=0
for asin in asin_map_needs_sentembedding:
    num_sents+=asin_map_needs_sentembedding[asin].shape[0]
    
#X_sentence=np.empty((1024,num_sents))
X_needs_sentence=np.empty((num_sents,1024))

i=0
for asin in asin_map_needs_sentembedding:
#    print(asin)#,asin_map_sentembedding[asin][:,1])
    num_sent=asin_map_needs_sentembedding[asin].shape[0]
#    X_sentence[:,i:i+num_sent]=np.transpose(asin_map_sentembedding[asin])
    X_needs_sentence[i:i+num_sent,:]=asin_map_needs_sentembedding[asin]
    i=i+num_sent

#---------------3.get the sent embeding after calculation------------------
embedding_size=1024

 # pad the vector?  (occurs if we have less sentences than embeddings_size)
 #otherwise the svd will not perform
if len(X_needs_sentence) < embedding_size:
    append_mat=np.zeros((embedding_size - len(X_needs_sentence),embedding_size)) 
    X_needs_sentence=np.concatenate((X_needs_sentence, append_mat), axis=0)

pca = PCA(n_components=embedding_size)
pca.fit(np.array(X_needs_sentence))
u = pca.components_[0]  # the PCA vector
u_ut = np.multiply(u, np.transpose(u))  # u x uT

    
asin_map_needs_sent_new={}
for asin in asin_map_needs_sentembedding:
    sents=asin_map_needs_sentembedding[asin]
    sents=sents-np.multiply(u_ut,sents)
    asin_map_needs_sent_new[asin]=sents


#---------------delete big variables for better memory usage--------------------------------------
del asin_map_needs


#------------use asin_map_needs_sent_new and asin_map_labels to finish experiments--------------------
#validate: half of asin_map_needs_sent_new
#test: half of asin_map_needs_sent_new

val_test_list=[]
for asin in asin_map_needs_sent_new:
    if(len(asin_map_needs_sent_new[asin])==1):
        val_test_list.append((asin_map_needs_sent_new[asin],asin_map_labels[asin]))
    else:
        for each in asin_map_needs_sent_new[asin]:
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









