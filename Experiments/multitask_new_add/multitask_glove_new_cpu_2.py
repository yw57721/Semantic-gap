# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:21:39 2019

@author: Li Xiang

changes made compared to elmo experiment:
    change asin_map_gene_reviews to glove embeddings

run 5 times, each time set cur_exp_param 
to get the y_train labels of review dataset

"""

import numpy as np
import sys
import func_elmo
import func_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd
from nltk.tokenize import word_tokenize

#--------------------------start parm setting--------------------------

cur_exp_param='screen'#['cpu','ram','hd','gpu','screen']
#cur_sent_embd_type='concat'#['max','ave','concat']

#----------------------------end parm setting-------------------------

if(cur_sent_embd_type=='concat'):
    cur_word_dimen=600
elif(cur_sent_embd_type=='max' or cur_sent_embd_type=='ave' ):
    cur_word_dimen=300


#------------prepare for length experiment-----------------------------

df=pd.read_csv("../data/generated_reviews.csv")
all_asins=list(df['asin'].unique())

asin_review=dict()

for _asin in all_asins: 
    asin_review[_asin]=df[df.asin==_asin]['reviews'].tolist()

#-----------------------------prepare train/test set----------------------------

if(cur_exp_param=='cpu'):   
    test_asin_list=func_elmo.get_new_cpu_useful_asin()
else:
    test_asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

if(cur_exp_param=='cpu'):   
    asin_map_labels=func_elmo.get_new_cpu_label()
else:
    asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

if(dir().count('embeddings_index')==0):
    embeddings_index = {}
    f = open('D:\Datasets\glove.6B.300d.txt',encoding='utf-8')
    
    for line in f:
        values = line.split(' ')
        word = values[0] ## The first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
        embeddings_index[word] = coefs
    f.close()
    
    print('GloVe data loaded')
    
asin_map_gene_reviews={}

"""
for _asin in test_asin_list:
    review_embd=func_elmo.get_gener_review_embedding(emb_type=cur_sent_embd_type,asin=_asin)
    if(review_embd is not None):
        asin_map_gene_reviews[_asin]=review_embd
        sent_length_lst.extend(func_elmo.get_review_length(asin_review,_asin))
        
"""

for asin in test_asin_list:
    asin_to_rev=df.loc[df['asin']==asin,'reviews'].tolist()
    if(asin_to_rev==[]):
        continue
    asin_rev_matrix=np.zeros((len(asin_to_rev),cur_word_dimen))
#    for each review
    for i,rev in enumerate(asin_to_rev):
        sent=np.zeros((1,cur_word_dimen))
        cur_word_toks=word_tokenize(rev)
        if(cur_sent_embd_type=='max'):
            reviews_mat=np.full((len(cur_word_toks),cur_word_dimen),-np.inf)
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    reviews_mat[j]=embeddings_index[w]
            sent=np.amax(reviews_mat, axis=0)  
            asin_rev_matrix[i]=sent
#            break
        if(cur_sent_embd_type=='ave'):
            reviews_mat=np.zeros((len(cur_word_toks),cur_word_dimen))
            div=0
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    div+=1
                    reviews_mat[j]=embeddings_index[w]
            sent=np.sum(reviews_mat, axis=0)/div
            asin_rev_matrix[i]=sent       
            
        if(cur_sent_embd_type=='concat'): #['max','ave','concat'])
            
            max_reviews_mat=np.full((len(cur_word_toks),300),-np.inf)
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    max_reviews_mat[j]=embeddings_index[w]
            max_sent=np.amax(max_reviews_mat, axis=0)  
            
            ave_reviews_mat=np.zeros((len(cur_word_toks),300))
            div=0
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    div+=1
                    ave_reviews_mat[j]=embeddings_index[w]
            ave_sent=np.sum(ave_reviews_mat, axis=0)/div
            asin_rev_matrix[i]=np.concatenate((max_sent,ave_sent))
            
    asin_map_gene_reviews[asin]=asin_rev_matrix 


"""
code afterwards is the same as the elmo experiment
"""

#append reviews embeddings into sample
sample = np.zeros(shape=(1,cur_word_dimen))

y=[]
for _asin in asin_map_gene_reviews:
    sample=np.concatenate((sample,asin_map_gene_reviews[_asin]),axis=0)
    for i in range(asin_map_gene_reviews[_asin].shape[0]):
        y.append(asin_map_labels[_asin])
sample=sample[1:]

#sys.exit(0)

X=sample
#change the random labels in y from 0 to N
if(y[0]!=[]):
    if(cur_exp_param=='cpu'):
        all_labels=sorted(list(set(np.array(y).transpose().tolist()[0])))
    else:
        all_labels=sorted(list(set(y[0])))
new_map_ylabel={}
for i in range(len(all_labels)):
    new_map_ylabel[all_labels[i]]=i
y_new=[]
for each_all_labels in y:
    y_new.append(list(map(lambda x : new_map_ylabel[x],each_all_labels)))
y_array=np.array(y_new)

#y_col_count is the total label count
y_col_count=y_array.shape[1]

Xgen_train, Xgen_test, ygen_train_all_labels, ygen_test_all_labels= \
    train_test_split(X, y_array, test_size=0.5, random_state=33)

# rank_indices: sort indices by length in Xgen_test
#rank_indices=[ind for (ind, r_len) in sorted(list(enumerate(gen_test_length_lst)),key=lambda x:x[1],reverse=True)]

#sys.exit(0)

ygen_train=np.array(ygen_train_all_labels)[:,0].tolist()
ygen_test=np.array(ygen_test_all_labels)[:,0].tolist()

ygen_val_cat = np_utils.to_categorical(ygen_train)
ygen_test_cat = np_utils.to_categorical(ygen_test)

try:
    ygen_val_dic
except NameError:
    ygen_val_dic={}
    
try:
    ygen_test_dic
except NameError:
    ygen_test_dic={}

#--record the correct label
try:
    ygen_test_all_label_dic
except NameError:
    ygen_test_all_label_dic={}

#----for experiment of label frequency
try:
    test_sort_labels
except NameError:
    test_sort_labels = {}
    
if((cur_exp_param=='ram')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

elif((cur_exp_param=='hd')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    
elif((cur_exp_param=='gpu')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    
elif((cur_exp_param=='screen')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    
elif((cur_exp_param=='cpu')):
    ygen_val_dic[cur_exp_param]=ygen_val_cat
    ygen_test_dic[cur_exp_param]=ygen_test_cat
    ygen_test_all_label_dic[cur_exp_param]=ygen_test_all_labels
    
    test_label_freq_dic={}
    for l in list(set(ygen_test)):
        test_label_freq_dic[l]=ygen_test.count(l)
    test_label_freq=list(test_label_freq_dic.items())
    test_sort_labels[cur_exp_param]=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

    












