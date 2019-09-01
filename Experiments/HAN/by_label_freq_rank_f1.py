# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:25:30 2019

@author: Li Xiang

run HAN first to get y_train

go to set_params part to set draw top 5 or top 10 results
"""

#------1.get label frequency of CPU and HD-------------------------------------

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
        
#------1.2. partitioned into different groups----------------------------------
label_map_group={}
group_num=5
for i,l in enumerate(new_sort_list):
    label_map_group[l]=int(i/group_num)
    
#------2.load model and give prediciton----------------------------------------

filepath='./models/'+cur_exp_param+'_model.hdf5'

if os.path.exists(filepath):
    model.load_weights(filepath)


test_predict_prob=model.predict(X_test) 
test_predict=np.argmax(test_predict_prob, axis=1)

test_predict_top_5=np.argsort(-test_predict_prob, axis=1)[:,:5]
test_predict_top_10=np.argsort(-test_predict_prob, axis=1)[:,:10]
test_predict_all=np.argsort(-test_predict_prob, axis=1)

labels_to_eval=y_test_all_labels[:,:math.floor(difference_percent*output_classes)]
print('relevant classes:',math.floor(difference_percent*output_classes))
print('total classes:',output_classes)

#-----set params(top 5 or top 10)-----------------------------------
test_predict_top_k=test_predict_top_5
#top_k=5

#------2.F1 value of each group------------------------------------------------

#f1_score=np.zeros(math.ceil(len(label_map_group)/group_num)*group_num)#(len(label_map_group)/group_num,5))
f1_score=np.zeros(len(label_map_group))#(len(label_map_group)/group_num,5))
recalls=np.zeros(len(label_map_group))
precisions=np.zeros(len(label_map_group))

for j,label in enumerate(label_map_group.keys()):
    indices = [i for i, l in enumerate(y_test) if l == label]
    each_label_real=[y_test[i] for i in indices]
    if(each_label_real==[]):
        f1_score[j]=0
        print(f1_score[j])
        continue
    
    each_label_predict=np.array([test_predict_top_k[i] for i in indices])
    each_labels_to_eval=np.array([labels_to_eval[i] for i in indices])
    precision=func_eval._precision_score(each_label_predict,each_labels_to_eval)
    precisions[j]=precision
    recall=func_eval.new_recall(each_label_predict,each_labels_to_eval)
    recalls[j]=recall
    if((precision+recall)!=0):
        f1_score[j]=precision*recall/(precision+recall)
    else:
        f1_score[j]=0
        
        
print('f1_score',f1_score)
#print('recalls',recalls)
#print('precisions',precisions)


#-----3.visulization of each group--------------------
    
    
    