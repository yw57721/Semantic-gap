# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:12:05 2019

@author: Li Xiang

"""


import sys
import numpy as np
import sys
import func_elmo
import func_eval
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
import math

#-------------------set params for different purpose--------------------

#cur_exp_param in ['cpu','ram','hd','gpu','screen']
#cur_sent_embd_type in ['max','ave','concat']

warm_start_set=True#True#False
cur_exp_param='hd'#['cpu','ram','hd','gpu','screen']
cur_sent_embd_type='max'#['max','ave','concat']
classifier_type='mlp'#'svm''mlp'
K=5 #top K results for evaluation
difference_percent=0.2 

#----------------------------end parm setting-------------------------

if(cur_sent_embd_type=='concat'):
    cur_word_dimen=2048
elif(cur_sent_embd_type=='max' or cur_sent_embd_type=='ave' ):
    cur_word_dimen=1024


#-----------------------------prepare train/test set----------------------------
asin_list=func_elmo.get_useful_asins(exp_param=cur_exp_param)
#sys.exit(0)

asin_map_reviews={}
for _asin in asin_list:
    asin_map_reviews[_asin]=func_elmo.get_sent_embedding(emb_type=cur_sent_embd_type,asin=_asin)

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
X_train, X_test, y_train_all_labels, y_test_all_labels = train_test_split(X, y_array, test_size=0.2, random_state=42)

#now we only take the first label(real label) to do classification
y_train=np.array(y_train_all_labels)[:,0].tolist()
y_test=np.array(y_test_all_labels)[:,0].tolist()


#---------------------------Start Training------------------------------------

if(classifier_type=='svm'):
#    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', C=1, probability=True, random_state=0))
    classifier = OneVsRestClassifier(linear_model.SGDClassifier(max_iter=500, tol=1e-3,random_state=21,
                                                                warm_start=warm_start_set))

if(classifier_type=='mlp'):
    classifier = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500, alpha=0.0001,
                     solver='adam', verbose=0,  random_state=21)
#                     warm_start=warm_start_set)



sys.exit(0)
print("Strat training...")
classifier.fit(X_train,y_train)
#sys.exit(0)

#---------------------use X_test for evaluation------------------

if(classifier_type=='svm'):
    #note that in svm predict_proba is inconsistent with predict function
    #use decision_function-->consistent
    y_pred_proba = classifier.decision_function(X_test)#return inverse of distance

if(classifier_type=='mlp'):
    y_pred_proba = classifier.predict_proba(X_test)
    
all_labels=classifier.classes_

#----------------------get top-k results-------------------------

#print("Training finished(test on original dataset):\ncomponent type: {} \nemddeing: {} \nclassifier: {}\n" \
#      .format(cur_exp_param,cur_sent_embd_type,classifier_type))

y_top_K=[]
# --pick out the max probability labels(by sorting predict_proba or decision_function)
#--note this may be different in rnn
if(classifier_type=='mlp' or classifier_type=='svm'):
    for each_proba in y_pred_proba:
        sort_proba_index = each_proba.argsort()
        #sort all_labels in descending order
        sorted_arr1 = all_labels[sort_proba_index[::-1]]
        y_top_K.append(sorted_arr1[:K])
y_top_K=np.array(y_top_K)

if(classifier_type=='rnn'):
    print("error! y_top_K not defined")

#----------------------calculate top-K PR value-------------------------
#print('ndcg:')
#i = 0
#ndcgs = []
#labels_to_eval=np.array(y_test_all_labels)[:,:math.ceil(difference_percent*y_col_count)]
#   
#while i < K:
#    
#    y_pred = y_top_K[:, 0:i+1]
#    i = i+1
#
#    ndcg_i = func_eval._NDCG_score(y_pred,labels_to_eval)
#    ndcgs.append(ndcg_i)
#
#    print(ndcg_i)
#
#
##sys.exit(0)  
#
#print("precision:")
#i = 0
#precisions = []
#labels_to_eval=np.array(y_test_all_labels)[:,:math.ceil(difference_percent*y_col_count)]
#  
#while i < K:
#    
#    y_pred = y_top_K[:, 0:i+1]
#    i = i+1
#
#    precision = func_eval._precision_score(y_pred,labels_to_eval)
#    precisions.append(precision)
#
#    print(precision)
#
#print("recall:")
#
#i = 0
#recalls = []
#while i < K:
#    
#    y_pred = y_top_K[:,  0:i+1]
#
#    i = i+1   
##    labels_to_eval=np.array(y_test_all_labels)[:,:math.ceil(difference_percent*y_col_count)]
#    recall = func_eval.new_recall_score(y_pred, labels_to_eval)
#    recalls.append(recall)
#
#    print(recall)
#
#
#
#
#
#
#
#
#








