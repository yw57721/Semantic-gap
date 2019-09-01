# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:00:13 2019

@author: Li Xiang

load the pretrained MLP model and evaluate by label's frequency in train set

*run glove_train first to get data set

"""
import numpy as np
import os
import sys
import func_eval

#--------------set params-------------------------------------------------
#set to 5 normally
top_k_to_eval=2#evaluate top_k results of each group

#set #groups of each label
label_map_group={}
if(cur_exp_param=='cpu'):
    group_num=5
elif(cur_exp_param=='ram'):
    group_num=3#6
elif(cur_exp_param=='hd'):
    group_num=3
elif(cur_exp_param=='screen'):
    group_num=3#5
elif(cur_exp_param=='gpu'):
    group_num=3#6
    
#--------------end setting------------------------------------------------

#--------1.sort labels in the train set by label frequency(descending)-------------------------
#----------based on the sorted results sort labels in test set

train_label_freq={}
for l in list(set(y_train)):
    train_label_freq[l]=y_train.count(l)
train_label_freq=list(train_label_freq.items())
train_sort_labels=[l for l,r in sorted(train_label_freq,key=lambda x:x[1],reverse=True)]

test_label_freq_dic={}
for l in list(set(y_test)):
    test_label_freq_dic[l]=y_test.count(l)
test_label_freq=list(test_label_freq_dic.items())
test_sort_labels=[l for l,r in sorted(test_label_freq,key=lambda x:x[1],reverse=True)]

#this filter out labels that not appear in test set
new_sort_list=[]
for label in train_sort_labels:
    if(label in test_sort_labels):
        new_sort_list.append(label)

all_test_classes=set(new_sort_list)

#------2.load MLP model and give prediciton----------------------------------------

filepath='./models/'+cur_exp_param+'_'+cur_sent_embd_type+'_model.pkl'

#load the MLP classifier based on exp_param and sent_emb_type
if os.path.exists(filepath):
    model = joblib.load(filepath)


test_predict_prob=model.predict_proba(X_test_emb)
test_predict=np.argmax(test_predict_prob, axis=1)

test_predict_top_k=np.argsort(-test_predict_prob, axis=1)[:,:top_k_to_eval]
test_predict_top_5=np.argsort(-test_predict_prob, axis=1)[:,:5]
test_predict_top_10=np.argsort(-test_predict_prob, axis=1)[:,:10]
test_predict_all=np.argsort(-test_predict_prob, axis=1)


if(cur_exp_param=='cpu'):
    labels_to_eval=y_test_all_labels[:,:math.floor(difference_percent*output_classes)]
else:
    labels_to_eval=y_test_all_labels[:,:1]

print('relevant classes:',math.floor(difference_percent*output_classes))
print('total classes in train set:',output_classes)
print('total classes in test set:',len(all_test_classes))

#-----set params(top 5 or top 10)-----------------------------------
if(cur_exp_param=='cpu'):
    test_predict_top_k=test_predict_top_10
else:
    test_predict_top_k=test_predict_top_k#test_predict_top_5#test_predict_top_5

#------2.evaluation of each group------------------------------------------------

recalls=np.zeros(len(all_test_classes))
precisions=np.zeros(len(all_test_classes))
ncdgs=np.zeros(len(all_test_classes))

#-----in test set, find the sample indices of each label---------------------
#-----then evaluate each label and append to recalls,ncdgs,precisions--------

for j,label in enumerate(new_sort_list):
    indices = [i for i, l in enumerate(y_test) if l == label]
    each_label_real=[y_test[i] for i in indices]
#    if(each_label_real==[]):
#        f1_score[j]=0
#        print(f1_score[j])
#        continue
    
    each_label_predict=np.array([test_predict_top_k[i] for i in indices])
    each_labels_to_eval=np.array([labels_to_eval[i] for i in indices])
    
    precision=func_eval._precision_score(each_label_predict,each_labels_to_eval)
    precisions[j]=precision
    recall=func_eval.new_recall(each_label_predict,each_labels_to_eval)
    recalls[j]=recall
    ncdg=func_eval._NDCG_score(each_label_predict,each_labels_to_eval)
    ncdgs[j]=ncdg
        
        
#print('\nrecalls:',recalls)
#print('precisions:',precisions)
#print('ncdgs:',ncdgs)

#----------3. split labels into group by label frequency-------------------------
#-----------get the evaluation of each group------------------------------

    
    
group_recalls=[]    
group_precisions=[]    
group_ncdgs=[]    


num_per_group=math.ceil(len(all_test_classes)/group_num)

for i in range(group_num):
    group_recalls.append(sum(recalls[num_per_group*i:num_per_group*(i+1)])/num_per_group)
    group_precisions.append(sum(precisions[num_per_group*i:num_per_group*(i+1)])/num_per_group)
    group_ncdgs.append(sum(ncdgs[num_per_group*i:num_per_group*(i+1)])/num_per_group)
    
    if((num_per_group*(i+2))>len(all_test_classes)):
        break
    
if(len(recalls[num_per_group*(i+1):])!=0):
    group_recalls.append(sum(recalls[num_per_group*(i+1):])/len(recalls[num_per_group*(i+1):]))
    group_precisions.append(sum(precisions[num_per_group*(i+1):])/len(precisions[num_per_group*(i+1):]))
    group_ncdgs.append(sum(ncdgs[num_per_group*(i+1):])/len(ncdgs[num_per_group*(i+1):]))

print("\ncur exp setting: {}, {}".format(cur_exp_param,cur_sent_embd_type))

print("Split into {} groups".format(group_num))

print('\nrecalls:')
for r in group_recalls:
    print(r)
    
#print('precisions:')#,group_precisions)
#for p in group_precisions:
#    print(p)

group_f1s=[]
for i in range(len(group_precisions)):
    f1=(group_precisions[i]*group_recalls[i])/(group_precisions[i]+group_recalls[i])
    group_f1s.append(f1)

print("f1:")
for f in group_f1s:
    print(f)
    
#print('ncdgs:')#,group_ncdgs)
#for n in group_ncdgs:
#    print(n)


