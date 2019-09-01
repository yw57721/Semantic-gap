# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:10:08 2019

@author: Li Xiang

run MR and MR_2 first to get train,val and test set
then run this file

"""
import func_eval
from sklearn.neural_network import MLPClassifier


hidden_neurons=256
    
classifier = MLPClassifier(hidden_layer_sizes=(hidden_neurons,hidden_neurons), max_iter=1000, alpha=0.001,
                     solver='adam', verbose=2,  random_state=21)

print("start training..")
classifier.fit(X_train,y_train)

new_coefs=classifier.coefs_[:2]

if(cur_exp_param=='cpu'):
#        new_coefs.append(np.random.rand(100, 23))
    new_coefs.append(np.zeros((100, 23)))
if(cur_exp_param=='ram'):
    new_coefs.append(np.random.rand(100, 6))
if(cur_exp_param=='hd'):
    new_coefs.append(np.random.rand(100, 11))
if(cur_exp_param=='gpu'):
    new_coefs.append(np.random.rand(100, 8))
if(cur_exp_param=='screen'):
    new_coefs.append(np.random.rand(100, 9))

finetune_classifier=MLPClassifier(hidden_layer_sizes=(hidden_neurons,hidden_neurons), max_iter=1000, alpha=0.001,
                     solver='adam', verbose=1,  random_state=21)

finetune_classifier.coefs_=new_coefs
finetune_classifier.fit(X_val,y_val)

y_pred_proba = finetune_classifier.predict_proba(X_test)

all_labels=finetune_classifier.classes_



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


if(cur_exp_param=='cpu'):
    labels_to_eval=y_test_all_labels[:,:4]
else:
    labels_to_eval=y_test_all_labels[:,:1]
#sys.exit(0)


print('\nndcg:')
i = 0
ndcgs = []


if(cur_exp_param=='cpu'):
    labels_to_eval=y_test_all_labels[:,:4]
else:
    labels_to_eval=y_test_all_labels[:,:1]
    
if(cur_exp_param=='cpu'):
    top_k=10
else:
    top_k=5
    
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


