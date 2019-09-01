# -*- coding: utf-8 -*-
"""
Elmo classification work based on WR algorithm(the sentence embedding)
    (A simple but tought-to-beat .....)

Created on Wed Mar 13 19:36:57 2019

@author: Li Xiang

this file apply MR to train set
"""
import sys
import func_elmo
import h5py
import os 
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

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
    total_word_list.extend(asin_map_review_words[_asin])

a=list(map(lambda x:x.split(),total_word_list))
#all words to lowercase
total_word_list=[item.lower() for sublist in a for item in sublist]
#total_word_list = [re.sub(r"[^A-Za-z]+", '', item).lower() for sublist in a for item in sublist]
total_word_len=len(total_word_list)
#sys.exit(0)
words_set=set(total_word_list)

words_map_frequency={}
for w in words_set:
    words_map_frequency[w]=total_word_list.count(w)/total_word_len

"""
------------compute sentence embeddings
#----------construct sentence embeddings based on ----------------------
#----------asin_map_review_words and asin_map_labels ---------------------

"""
#------------------1.compute sent embeddings---------------------------
a=10**-4

asin_map_sentembedding={}

for asin in asin_map_review_words:
#    print("{}".format(asin))
    length=len(asin_map_review_words[asin])
    sent_embeds=np.zeros((length,1024))
    if(asin_map_reviews[asin] is None):
        continue
    for i,sent in enumerate(asin_map_review_words[asin]):
        sent_embed=np.zeros((1024,))
        for j,w in enumerate(sent.split()):
            
            sent_embed+=a/(a+words_map_frequency[w.lower()])*\
                        asin_map_reviews[asin][i][j]
        sent_embeds[i]=sent_embed
    asin_map_sentembedding[asin]=sent_embeds
#    print("{} finished!".format(asin))

print("1 finished")
#------------------2.form the matrix X_sentence(cols are sent embeddings)---------------------------
num_sents=0
for asin in asin_map_sentembedding:
    num_sents+=asin_map_sentembedding[asin].shape[0]
    
#X_sentence=np.empty((1024,num_sents))
X_sentence=np.empty((num_sents,1024))

i=0
for asin in asin_map_sentembedding:
#    print(asin)#,asin_map_sentembedding[asin][:,1])
    num_sent=asin_map_sentembedding[asin].shape[0]
#    X_sentence[:,i:i+num_sent]=np.transpose(asin_map_sentembedding[asin])
    X_sentence[i:i+num_sent,:]=asin_map_sentembedding[asin]
    i=i+num_sent
    
#---------------3.get the sent embeding after calculation------------------
embedding_size=1024

pca = PCA(n_components=embedding_size)
pca.fit(np.array(X_sentence))
u = pca.components_[0]  # the PCA vector
u_ut = np.multiply(u, np.transpose(u))  # u x uT

 # pad the vector?  (occurs if we have less sentences than embeddings_size)
if len(u) < embedding_size:
    for i in range(embedding_size - len(u)):
        u = np.append(u, 0)  # add needed extension for multiplication below


asin_map_sentembedding_new={}
for asin in asin_map_sentembedding:
    sents=asin_map_sentembedding[asin]
    sents=sents-np.multiply(u_ut,sents)
    asin_map_sentembedding_new[asin]=sents

#---------------delete big variables for better memory usage--------------------------------------
del asin_map_reviews

#------------use asin_map_sentembedding_new and asin_map_labels to ---------
#------------train the classifier first ------------------------------------
#---join the training sample based on asin first----------------------------
train_list=[]
for asin in asin_map_sentembedding_new:
    if(len(asin_map_sentembedding_new[asin])==1):
        train_list.append((asin_map_sentembedding_new[asin],asin_map_labels[asin]))
    else:
        for each in asin_map_sentembedding_new[asin]:
            train_list.append((each,asin_map_labels[asin]))


embded_size=1024

X_train=np.zeros((len(train_list),embded_size))
y_train_all_labels=[]
y_train=[]

for i,s in enumerate(train_list):
    X_train[i]=s[0]
    y_train_all_labels.append(s[1])
    y_train.append(s[1][0])


#X_train,y_train








