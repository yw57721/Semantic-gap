# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:22:45 2019

@author: Li Xiang

run HAN first before run this file to get X_test,y_test_all_labels and so on 

"""
import numpy as np


#---------set params----------------------------
exp_param='long'#'long'#'short'

top_k=10#10#5

#--------param setting ends----------------------

long_short_split=0.5
test_reviews_len=[]
for r in X_test:
    test_reviews_len.append(np.nonzero(r)[0].shape[0])


rank_indices=[ind for (ind, r_len) in sorted(list(enumerate(test_reviews_len)),key=lambda x:x[1],reverse=True)]

long_rev_inds=rank_indices[:int(len(rank_indices)*long_short_split)]
short_rev_inds=rank_indices[int(len(rank_indices)*long_short_split):]

X_test_long=np.array([X_test[i] for i in long_rev_inds])
X_test_short=np.array([X_test[i] for i in short_rev_inds])

y_test_all_labels_long=np.array([y_test_all_labels[i] for i in long_rev_inds])
y_test_all_labels_short=np.array([y_test_all_labels[i] for i in short_rev_inds])

##---------------load pretrained model----------------------------------------

filepath='./models/'+cur_exp_param+'_model.hdf5'

if os.path.exists(filepath):
    model.load_weights(filepath)


#-------predict long reviews---------------------------------
print('relevant classes:',math.floor(difference_percent*output_classes))
print('total classes:',output_classes)

if(exp_param=='long'):
    test_predict_prob=model.predict(X_test_long) 
    test_predict=np.argmax(test_predict_prob, axis=1)
    
    test_predict_top_5=np.argsort(-test_predict_prob, axis=1)[:,:5]
    test_predict_top_10=np.argsort(-test_predict_prob, axis=1)[:,:10]
    test_predict_all=np.argsort(-test_predict_prob, axis=1)

    labels_to_eval=y_test_all_labels_long[:,:math.floor(difference_percent*output_classes)]


#-------predict short reviews---------------------------------

if(exp_param=='short'):
    test_predict_prob=model.predict(X_test_short) 
    test_predict=np.argmax(test_predict_prob, axis=1)
    
    test_predict_top_5=np.argsort(-test_predict_prob, axis=1)[:,:5]
    test_predict_top_10=np.argsort(-test_predict_prob, axis=1)[:,:10]
    test_predict_all=np.argsort(-test_predict_prob, axis=1)
    
    labels_to_eval=y_test_all_labels_short[:,:math.floor(difference_percent*output_classes)]


test_predict_top_k=test_predict_top_10

print("\ncur_exp_setting: {}, {}".format(cur_exp_param,exp_param))

print('\nndcg:')
i = 0
ndcgs = []
if(math.floor(difference_percent*output_classes)==0):
    print("error: the relevant classes number is 0, \
          you may want to consider math.ceiling instead of math.floor\
          ")

while i < top_k:
    
    y_pred = test_predict_top_k[:, 0:i+1]
    i = i+1

    ndcg_i = func_eval._NDCG_score(y_pred,labels_to_eval)
    ndcgs.append(ndcg_i)

    print(ndcg_i)


#sys.exit(0)  

print("precision:")
i = 0
precisions = []

while i < top_k:
    
    y_pred = test_predict_top_k[:, 0:i+1]
    i = i+1

    precision = func_eval._precision_score(y_pred,labels_to_eval)
    precisions.append(precision)

    print(precision)

print("recall:")
i = 0
recalls = []
while i < top_k:
    
    y_pred = test_predict_top_k[:,  0:i+1]

    i = i+1   
    recall = func_eval.new_recall(y_pred, labels_to_eval)
    recalls.append(recall)

    print(recall)



