# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:27:06 2019

@author: Li Xiang

load the pretrained MLP and test by review length (long / short group)
*run glove_train first to prepare data set ready

"""

import numpy as np
from sklearn.externals import joblib


#---------set params----------------------------
exp_param='short'#'long'#'short'
long_short_split=0.2 # long top 20% longest

#--------param setting ends----------------------

test_reviews_len=[]
for r in X_test:
    test_reviews_len.append(np.nonzero(r)[0].shape[0])


rank_indices=[ind for (ind, r_len) in sorted(list(enumerate(test_reviews_len)),key=lambda x:x[1],reverse=True)]

long_rev_inds=rank_indices[:int(len(rank_indices)*long_short_split)]
short_rev_inds=rank_indices[int(len(rank_indices)*long_short_split):]

X_test_long=np.array([X_test_emb[i] for i in long_rev_inds])
X_test_short=np.array([X_test_emb[i] for i in short_rev_inds])

y_test_all_labels_long=np.array([y_test_all_labels[i] for i in long_rev_inds])
y_test_all_labels_short=np.array([y_test_all_labels[i] for i in short_rev_inds])


#----------load model and give predictions------------------
model_name='./models/'+cur_exp_param+'_'+cur_sent_embd_type+'_model.pkl'
load_classifier = joblib.load(model_name) 

if(exp_param=='long'):
    y_pred_proba = load_classifier.predict_proba(X_test_long)
if(exp_param=='short'):
    y_pred_proba = load_classifier.predict_proba(X_test_short)

all_labels=load_classifier.classes_


if(cur_exp_param=='cpu'):
    top_k=10
else:
    top_k=5
    
y_top_K=[]
#--pick out the max probability labels(by sorting predict_proba or decision_function)
#--note this may be different in rnn
for each_proba in y_pred_proba:
    sort_proba_index = each_proba.argsort()
    #sort all_labels in descending order
    sorted_arr1 = all_labels[sort_proba_index[::-1]]
    y_top_K.append(sorted_arr1[:top_k])
y_top_K=np.array(y_top_K)
test_predict_top_k=y_top_K


if(exp_param=='long'):
    y_test_all_labels_to_use = y_test_all_labels_long
if(exp_param=='short'):
    y_test_all_labels_to_use = y_test_all_labels_short


if(cur_exp_param=='cpu'):
    labels_to_eval=y_test_all_labels_to_use[:,:math.floor(difference_percent*output_classes)]
else:
    labels_to_eval=y_test_all_labels_to_use[:,:1]
#sys.exit(0)
print("\ncur_exp_setting: {}, {},{}".format(cur_exp_param,exp_param,cur_sent_embd_type))


#print('\nndcg:')
#i = 0
#ndcgs = []
#
#while i < top_k:
#    
#    y_pred = test_predict_top_k[:, 0:i+1]
#    i = i+1
#
#    ndcg_i = func_eval._NDCG_score(y_pred,labels_to_eval)
#    ndcgs.append(ndcg_i)
#
#    print(ndcg_i)


#sys.exit(0)  

#print("precision:")
i = 0
precisions = []

while i < top_k:
    
    y_pred = test_predict_top_k[:, 0:i+1]
    i = i+1

    precision = func_eval._precision_score(y_pred,labels_to_eval)
    precisions.append(precision)

#    print(precision)

print("\nrecall:")
i = 0
recalls = []
while i < top_k:
    
    y_pred = test_predict_top_k[:,  0:i+1]

    i = i+1   
    recall = func_eval.new_recall(y_pred, labels_to_eval)
    recalls.append(recall)

    print(recall)

print("f1 score:")
f1s=[]
for i in range(len(recalls)):
    f1=(recalls[i]*precisions[i])/(recalls[i]+precisions[i])
    f1s.append(f1)
    print(f1)

#-------record for drawing------
long_recalls=[]
short_recalls=[]
long_f1s=[]
short_f1s=[]

if(exp_param=='short'):
    short_recalls=recalls
    short_f1s=f1s
elif(exp_param=='long'):
    long_recalls=recalls
    long_f1s=f1s