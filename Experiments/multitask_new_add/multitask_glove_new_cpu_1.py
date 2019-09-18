# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:38:54 2019

@author: Li Xiang

new multitask learning:
    relabel the cpu label to 1-10 and include cpu into multitask training
    using glove embeddings

changes made compared to elmo experiment:
    only modified asin_map_reviews to glove sentence embedding

run 5 times, each time set different cur_exp_param 
 get the y_train labels of review dataset

cpu label mapping table:
    Description	                Label
    Intel Celeron/ADM A (0, 2GHz)	0
    Intel Celeron/ADM A ([2, 3)GHz	1
    Intel Celeron/ADM A ([3, )GHz	2
    Intel i3 (0, 2.4) GHz	         3
    Intel i3 [2.4, ) GHz	         4
    Intel i5 (0, 2] GHz	            5
    Intel i5 (2, 3) GHz	            6
    Intel i5 [3, ) GHz	            7
    Intel i7 (0, 2] GHz	            6
    Intel i7 (2, 3] GHz	            7
    Intel i7 [3, ) GHz	            8
    Others	                        9    
"""

import sys
import numpy as np
import func_elmo
import math
import func_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd
from nltk.tokenize import word_tokenize

#-------------------set params for different purpose--------------------

warm_start_set=True#True#False
cur_exp_param='screen' #['cpu','ram','hd','gpu','screen']
cur_sent_embd_type='concat' #['max','ave','concat']
#classifier_type='svm'#'svm''mlp'
K=5 #top K results for evaluation
difference_percent=0.2


#--------------------------end parm setting-------------------------

if(cur_sent_embd_type=='concat'):
    cur_word_dimen=600
elif(cur_sent_embd_type=='max' or cur_sent_embd_type=='ave' ):
    cur_word_dimen=300

#-----------------------------prepare train/test set----------------------------

if(cur_exp_param=='cpu'):   
    asin_list=func_elmo.get_new_cpu_useful_asin()
else:
    asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)


#sys.exit(0)

if(cur_exp_param=='cpu'):   
    asin_map_labels=func_elmo.get_new_cpu_label()
else:
    asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

if(dir().count('data_train')==0):
    df=pd.read_csv("../data/original_dataset.csv",encoding = "ISO-8859-1")
    df=df[df.asin.str.startswith('B')]
    df=df.reset_index(drop=True)
    df = df.rename(columns={'reviews': 'review'})
    df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
    
#review_lst=df['review'].tolist()
#review_token_lst=[word_tokenize(r) for r in review_lst]

#----------load glove embeddings into memory------------------------------- 
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

#-----------get the glove embedding of review and feed into asin_map_reviews

asin_map_reviews={}

for asin in asin_list:
    asin_to_rev=df.loc[df['asin']==asin,'review'].tolist()
    asin_rev_matrix=np.zeros((len(asin_to_rev),cur_word_dimen))
#    for each review
    for i,rev in enumerate(asin_to_rev):
        sent=np.zeros((1,cur_word_dimen))
        cur_word_toks=word_tokenize(rev)
        if(cur_sent_embd_type=='max'):
            reviews_mat=np.full((len(cur_word_toks),cur_word_dimen),-np.inf)
            flag=False
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    flag=True
                    reviews_mat[j]=embeddings_index[w]
            if(flag==True):
                sent=np.amax(reviews_mat, axis=0)  
                asin_rev_matrix[i]=sent
            else:
                asin_rev_matrix[i]=np.zeros((1,300))
#            break
        if(cur_sent_embd_type=='ave'):
            reviews_mat=np.zeros((len(cur_word_toks),cur_word_dimen))
            div=0
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    div+=1
                    reviews_mat[j]=embeddings_index[w]
            if(div!=0):
                sent=np.sum(reviews_mat, axis=0)/div
                asin_rev_matrix[i]=sent       
            
        if(cur_sent_embd_type=='concat'): #['max','ave','concat'])
            
            max_reviews_mat=np.full((len(cur_word_toks),300),-np.inf)
            flag=False
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    flag=True
                    max_reviews_mat[j]=embeddings_index[w]
            if(flag==True):
                max_sent=np.amax(max_reviews_mat, axis=0) 
            else:
                max_sent=np.zeros((300,))
            
            ave_reviews_mat=np.zeros((len(cur_word_toks),300))
            div=0
            for j,w in enumerate(cur_word_toks):
                if(w in embeddings_index):
                    div+=1
                    ave_reviews_mat[j]=embeddings_index[w]
            if(div!=0):
                ave_sent=np.sum(ave_reviews_mat, axis=0)/div
            else:
                ave_sent=np.zeros((300,))
            
            asin_rev_matrix[i]=np.concatenate((max_sent,ave_sent),axis=0)
            
    asin_map_reviews[asin]=asin_rev_matrix 

print('asin_map_reviews finished!' )

"""
code afterwards is the same as the elmo experiment
"""

#append reviews embeddings into sample
sample = np.zeros(shape=(1,cur_word_dimen))

y=[]
for _asin in asin_map_reviews:
    sample=np.concatenate((sample,asin_map_reviews[_asin]),axis=0)
    for i in range(asin_map_reviews[_asin].shape[0]):
        y.append(asin_map_labels[_asin])
sample=sample[1:]

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
num_classes=len(set(y_array[0]))



# y_XXX_all_labels stores the label rank results of each review
X_train, X_test, y_train_all_labels, y_test_all_labels = train_test_split(X, y_array, test_size=0.3, random_state=2)

#now we only take the first label(real label) to do classification
y_train=np.array(y_train_all_labels)[:,0].tolist()
y_test=np.array(y_test_all_labels)[:,0].tolist()

# encode class values as integers
y_train_cat = np_utils.to_categorical(y_train)
y_test_cat = np_utils.to_categorical(y_test)

try:
    y_train_dic
except NameError:
    y_train_dic = {}
    
try:
    y_test_dic
except NameError:
    y_test_dic = {}

#----for experiment of label frequency
try:
    train_sort_labels
except NameError:
    train_sort_labels = {}
    
    
if((cur_exp_param=='ram')):
    y_train_dic[cur_exp_param]=y_train_cat
    y_test_dic[cur_exp_param]=y_test_cat
#----for experiment of label frequency
    train_label_freq={}
    for l in list(set(y_train)):
        train_label_freq[l]=y_train.count(l)
    train_label_freq=list(train_label_freq.items())
    train_sort_labels[cur_exp_param]=[l for l,r in sorted(train_label_freq,key=lambda x:x[1],reverse=True)]

elif((cur_exp_param=='hd')):
    y_train_dic[cur_exp_param]=y_train_cat
    y_test_dic[cur_exp_param]=y_test_cat
#----for experiment of label frequency    
    train_label_freq={}
    for l in list(set(y_train)):
        train_label_freq[l]=y_train.count(l)
    train_label_freq=list(train_label_freq.items())
    train_sort_labels[cur_exp_param]=[l for l,r in sorted(train_label_freq,key=lambda x:x[1],reverse=True)]

elif((cur_exp_param=='gpu')):
    y_train_dic[cur_exp_param]=y_train_cat
    y_test_dic[cur_exp_param]=y_test_cat
#----for experiment of label frequency
    train_label_freq={}
    for l in list(set(y_train)):
        train_label_freq[l]=y_train.count(l)
    train_label_freq=list(train_label_freq.items())
    train_sort_labels[cur_exp_param]=[l for l,r in sorted(train_label_freq,key=lambda x:x[1],reverse=True)]

elif((cur_exp_param=='screen')):
    y_train_dic[cur_exp_param]=y_train_cat
    y_test_dic[cur_exp_param]=y_test_cat
#----for experiment of label frequency
    train_label_freq={}
    for l in list(set(y_train)):
        train_label_freq[l]=y_train.count(l)
    train_label_freq=list(train_label_freq.items())
    train_sort_labels[cur_exp_param]=[l for l,r in sorted(train_label_freq,key=lambda x:x[1],reverse=True)]

elif((cur_exp_param=='cpu')):
    y_train_dic[cur_exp_param]=y_train_cat
    y_test_dic[cur_exp_param]=y_test_cat
#----for experiment of label frequency
    train_label_freq={}
    for l in list(set(y_train)):
        train_label_freq[l]=y_train.count(l)
    train_label_freq=list(train_label_freq.items())
    train_sort_labels[cur_exp_param]=[l for l,r in sorted(train_label_freq,key=lambda x:x[1],reverse=True)]





















