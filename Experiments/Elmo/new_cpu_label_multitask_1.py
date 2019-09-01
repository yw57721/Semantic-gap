# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:13:28 2019

@author: Li Xiang

new multitask learning:
    relabel the cpu label to 1-10 and include cpu into multitask training

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

#-------------------set params for different purpose--------------------

warm_start_set=True#True#False
cur_exp_param='screen' #['cpu','ram','hd','gpu','screen']
cur_sent_embd_type='concat' #['max','ave','concat']
#classifier_type='svm'#'svm''mlp'
K=5 #top K results for evaluation
difference_percent=0.2

#--------------------------end parm setting-------------------------

if(cur_sent_embd_type=='concat'):
    cur_word_dimen=2048
elif(cur_sent_embd_type=='max' or cur_sent_embd_type=='ave' ):
    cur_word_dimen=1024


#-----------------------------prepare train/test set----------------------------

if(cur_exp_param=='cpu'):   
    asin_list=func_elmo.get_new_cpu_useful_asin()
else:
    asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)


#sys.exit(0)

asin_map_reviews={}
for _asin in asin_list:
    asin_map_reviews[_asin]=func_elmo.get_sent_embedding(emb_type=cur_sent_embd_type,asin=_asin)

if(cur_exp_param=='cpu'):   
    asin_map_labels=func_elmo.get_new_cpu_label()
else:
    asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)


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







