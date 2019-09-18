# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:46:04 2019

@author: Li Xiang

cpu relabeled to 0-9
run train_elmo_new_cpu_label.py first(get original model) 
and then run this file(finetune and validate)

"""

import sys
import numpy as np
import sys
import h5py
import func_elmo
import func_eval
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
import math
from copy import deepcopy

#-------------------set params for different purpose--------------------

#cur_exp_param in ['cpu','ram','hd','gpu','screen']
#cur_sent_embd_type in ['max','ave','concat']

#hd,gpu,screen not directly available 
#cur_exp_param='cpu'#['cpu','ram','hd','gpu','screen']
#cur_sent_embd_type='max'#['max','ave','concat']
#classifier_type='mlp'#'svm''mlp''rnn'
#K=5 #top K results for evaluation
#difference_percent=0.2

#----------------------------end parm setting-------------------------

if(cur_sent_embd_type=='concat'):
    cur_word_dimen=2048
elif(cur_sent_embd_type=='max' or cur_sent_embd_type=='ave' ):
    cur_word_dimen=1024

#-----------------------------prepare train/test set----------------------------
if(cur_exp_param=='cpu'):   
    test_asin_list=func_elmo.get_new_cpu_useful_asin()
else:
    test_asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

#test_asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)

asin_map_gene_reviews={}
for _asin in test_asin_list:
    review_embd=func_elmo.get_gener_review_embedding(emb_type=cur_sent_embd_type,asin=_asin)
    if(review_embd is not None):
        asin_map_gene_reviews[_asin]=review_embd
        
if(cur_exp_param=='cpu'):   
    asin_map_labels=func_elmo.get_new_cpu_label()
else:
    asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

#asin_map_labels=func_elmo.get_sorted_label_list(exp_param=cur_exp_param)

#sys.exit(0)

#append reviews embeddings into sample
sample = np.zeros(shape=(1,cur_word_dimen))

y=[]
for _asin in asin_map_gene_reviews:
    sample=np.concatenate((sample,asin_map_gene_reviews[_asin]),axis=0)
    for i in range(asin_map_gene_reviews[_asin].shape[0]):
        y.append(asin_map_labels[_asin])
sample=sample[1:]

X=sample
#change the random labels in y from 0 to N
#if(y[0]!=[]):
#    all_labels=sorted(list(set(y[0])))
#new_map_ylabel={}
#for i in range(len(all_labels)):
#    new_map_ylabel[all_labels[i]]=i
#y_new=[]
#for each_all_labels in y:
#    y_new.append(list(map(lambda x : new_map_ylabel[x],each_all_labels)))
y_array=np.array(y)

#y_col_count is the total label count
y_col_count=y_array.shape[1]

Xgen_train, Xgen_test, ygen_train_all_labels, ygen_test_all_labels = \
    train_test_split(X, y_array, test_size=0.5, random_state=30)

ygen_train=np.array(ygen_train_all_labels)[:,0].tolist()
ygen_test=np.array(ygen_test_all_labels)[:,0].tolist()

#sys.exit(0)
#set(y_top_K[:,0])

#--------finetune based on trained model 
if(classifier_type=='mlp'):      
    finetune_classifier=MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500, alpha=0.0001,
                     solver='adam', verbose=0,  random_state=21)
if(classifier_type=='svm'):      
#    finetune_classifier=OneVsRestClassifier(svm.SVC(kernel='linear', C=1, probability=True, random_state=0))
    finetune_classifier = OneVsRestClassifier(linear_model.SGDClassifier(max_iter=500, tol=1e-3,random_state=21))


#get the original classifier's hidden layer's params 
#and get rid of the classification layer's param
#refer to this url: https://stackoverflow.com/questions/45134834/fine-tuning-vs-retraining
if(classifier_type=='mlp'):      
    new_coefs=classifier.coefs_[:2]
    if(cur_exp_param=='cpu'):
#        new_coefs.append(np.random.rand(100, 23))
        new_coefs.append(np.zeros((100, 10)))#change here
    if(cur_exp_param=='ram'):
        new_coefs.append(np.random.rand(100, 6))
    if(cur_exp_param=='hd'):
        new_coefs.append(np.random.rand(100, 11))
    if(cur_exp_param=='gpu'):
        new_coefs.append(np.random.rand(100, 8))
    if(cur_exp_param=='screen'):
        new_coefs.append(np.random.rand(100, 9))
        
        
    finetune_classifier.coefs_=new_coefs
    finetune_classifier.fit(Xgen_train,ygen_train)

#copy the params, retrain by 50% generated data to finetune the params
if(classifier_type=='svm'):     
#    finetune_classifier.coef_=classifier.coef_
    finetune_classifier=deepcopy(classifier)
    finetune_classifier.fit(Xgen_train,ygen_train)
    
#---------------finetune end---------------

if(classifier_type=='svm'):
    #note that in svm predict_proba is inconsistent with predict function
    #use decision_function-->consistent
    y_pred_proba = finetune_classifier.decision_function(Xgen_test)#return inverse of distance

if(classifier_type=='mlp'):
    y_pred_proba = finetune_classifier.predict_proba(Xgen_test)
    
all_labels=finetune_classifier.classes_


if(cur_exp_param=='cpu'):
    K=5
y_top_K=[]
#--pick out the max probability labels(by sorting predict_proba or decision_function)
#--note this may be different in rnn
if(classifier_type=='mlp' or classifier_type=='svm'):
    for each_proba in y_pred_proba:
        sort_proba_index = each_proba.argsort()
        #sort all_labels in descending order
        sorted_arr1 = all_labels[sort_proba_index[::-1]]
        y_top_K.append(sorted_arr1[:K])
y_top_K=np.array(y_top_K)

#sys.exit(0)


print("\ncomponent type: {} \nemddeing: {} \nclassifier: {}\n"\
      .format(cur_exp_param,cur_sent_embd_type,classifier_type))

print('ndcg:')
i = 0
ndcgs = []
labels_to_eval=np.array(ygen_test_all_labels)#[:,:math.ceil(difference_percent*y_col_count)]
   
if(cur_exp_param=='cpu'):
    K=5
    labels_to_eval=np.array(ygen_test_all_labels)[:,:1]
else:
#    this means we only use the top 1 as the correct label
    labels_to_eval=np.array(ygen_test_all_labels)[:,:1]
    
while i < K:
    
    y_pred = y_top_K[:, 0:i+1]
    i = i+1

    ndcg_i = func_eval._NDCG_score(y_pred,labels_to_eval)
    ndcgs.append(ndcg_i)

    print(ndcg_i)


print("precision:")
i = 0
precisions = []
labels_to_eval=np.array(ygen_test_all_labels)[:,:math.ceil(difference_percent*y_col_count)]
    
while i < K:
    
    y_pred = y_top_K[:, 0:i+1]
    i = i+1

    precision = func_eval._precision_score(y_pred,labels_to_eval)
    precisions.append(precision)

    print(precision)

print("recall:")

i = 0
recalls = []
while i < K:
    
    y_pred = y_top_K[:,  0:i+1]

    i = i+1   
#    labels_to_eval=np.array(y_test_all_labels)[:,:math.ceil(difference_percent*y_col_count)]
    recall = func_eval.new_recall(y_pred, labels_to_eval)
    recalls.append(recall)

    print(recall)













