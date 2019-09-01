# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:00:13 2019

@author: Li Xiang

load the pretrained MLP and test on the whole test set
*run glove_train first to get data set ready

"""
import func_eval

model_name='./models/'+cur_exp_param+'_'+cur_sent_embd_type+'_model.pkl'
load_classifier = joblib.load(model_name) 
print("\ncur_exp_setting: {},{}".format(cur_exp_param,cur_sent_embd_type))

#----------the code afterwards is to use the whole test dataset-----------

y_pred_proba = load_classifier.predict_proba(X_test_emb)

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


if(cur_exp_param=='cpu'):
    labels_to_eval=y_test_all_labels[:,:math.floor(difference_percent*output_classes)]
#    labels_to_eval=y_test_all_labels[:,:4]
else:
    labels_to_eval=y_test_all_labels[:,:1]
#sys.exit(0)


print('\nndcg:')
i = 0
ndcgs = []


#if(cur_exp_param=='cpu'):
#    labels_to_eval=y_test_all_labels[:,:math.floor(difference_percent*output_classes)]
#else:
#    labels_to_eval=y_test_all_labels[:,:1]
    
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


