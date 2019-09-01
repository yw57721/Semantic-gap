# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:55:01 2019

@author: Li Xiang

*run these 2files 4 times:(each with different types)

    run multitask_1 to get review (trainset) y_labels for 4 types
    
    run multitask_2 to get needs data (val, test) y_labels for 4 types

variables for model training:
    X_train, X_test
    y_train_dic, y_test_dic 
    
variables for model val/testing:
    Xgen_train, Xgen_test
    ygen_val_dic, ygen_test_dic

variables for label frequency:
    train_sort_labels,test_sort_labels,new_sort_list
"""

import sys
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import func_eval
import math

#------split the labels into 2 parts , frequent and unfrequent
# --------get frequency of each label and rerank---------------------------

#this filter out labels that not appear in test set
new_sort_list={}
for key in train_sort_labels:
    new_sort_list[key]=[]
    for label in train_sort_labels[key]:
        if(label in test_sort_labels[key]):
            new_sort_list[key].append(label)

#----------------------------------end--------------------------------------

#------------------------------setting params-------------------------------
            
top_k=3

#-------------------keras multitask model--------------------------------------
label_count={
    'gpu':8,
    'ram':6,
    'hd':11,
    'screen':9,
    'cpu':10
}
if(cur_sent_embd_type=='concat'):
    embed_dimen=2048#2048#1024
else:
    embed_dimen=1024#2048#1024
    
inputs = Input((embed_dimen,))
x = Dense(100, activation='relu')(inputs)
#x = Dense(100, activation='relu')(x)
x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()] 
model = Model(inputs, x) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

Y_train=[y_train_dic['gpu'],y_train_dic['ram'],y_train_dic['hd'],y_train_dic['screen'],y_train_dic['cpu']]
Y_test=[y_test_dic['gpu'],y_test_dic['ram'],y_test_dic['hd'],y_test_dic['screen'],y_test_dic['cpu']]
Ygen_val=[ygen_val_dic['gpu'],ygen_val_dic['ram'],ygen_val_dic['hd'],ygen_val_dic['screen'],ygen_val_dic['cpu']]
Ygen_test=[ygen_test_dic['gpu'],ygen_test_dic['ram'],ygen_test_dic['hd'],ygen_test_dic['screen'],ygen_test_dic['cpu']]

#model.fit(x=X_train,y=Y_train,validation_data=(Xgen_train,Ygen_val),epochs=15)
model.fit(x=X_train,y=Y_train,validation_data=(X_test,Y_test),epochs=15)
print('\n')
model.fit(Xgen_train,Ygen_val,validation_split=0.2,epochs=15)

y_pred_prob=model.predict(Xgen_test)
# y_classes is the top_k predicted results of each type
y_classes={}
for prediction,types in zip(y_pred_prob,list(label_count.keys())):
    y_classes[types] = prediction.argsort()[:,::-1][:,:top_k]


#---------------------------evaluation------------------------------ 
recalls={}#np.zeros(len(all_test_classes))
precisions={}#np.zeros(len(all_test_classes))
ncdgs={}#np.zeros(len(all_test_classes))

for key in y_classes:
    recalls[key]=[]
    precisions[key]=[]
    ncdgs[key]=[]
    
    labels_to_eval=ygen_test_all_label_dic[key][:,:1]
    test_predict_top_k=y_classes[key]
    
#-----evaluate by frequency------
    for j,label in enumerate(new_sort_list[key]):
        indices = [i for i, l in enumerate(ygen_test_all_label_dic[key][:,:1]) if l == label]
        
        each_label_predict=np.array([test_predict_top_k[i] for i in indices])
        each_labels_to_eval=np.array([labels_to_eval[i] for i in indices])
    
        precision=func_eval._precision_score(each_label_predict,each_labels_to_eval)
        precisions[key].append(precision)
        recall=func_eval.new_recall(each_label_predict,each_labels_to_eval)
        recalls[key].append(recall)
        ncdg=func_eval._NDCG_score(each_label_predict,each_labels_to_eval)
        ncdgs[key].append(ncdg)
        
print('recalls:',recalls)
print('precisions:',precisions)


#----------3. split labels into groups by label frequency-------------------------
#-----------get the evaluation of each group------------------------------

    
    
group_recalls={}  
group_precisions={}    
group_ncdgs={}  
group_f1s={}
for key in recalls:
    
    group_recalls[key]=[]
    group_precisions[key]=[]
    group_ncdgs[key]=[]
    
    if(key=='cpu'):
        group_num=3
    elif(key=='ram'):
        group_num=3#6
    elif(key=='hd'):
        group_num=3
    elif(key=='screen'):
        group_num=3#5
    elif(key=='gpu'):
        group_num=3#6
    
    num_per_group=math.ceil(len(recalls[key])/group_num)
    
    for i in range(group_num):
        group_recalls[key].append(sum(recalls[key][num_per_group*i:num_per_group*(i+1)])/num_per_group)
        group_precisions[key].append(sum(precisions[key][num_per_group*i:num_per_group*(i+1)])/num_per_group)
        group_ncdgs[key].append(sum(ncdgs[key][num_per_group*i:num_per_group*(i+1)])/num_per_group)
        
        if((num_per_group*(i+2))>len(recalls[key])):
            break
        
    if(len(recalls[key][num_per_group*(i+1):])!=0):
        group_recalls[key].append(sum(recalls[key][num_per_group*(i+1):])/len(recalls[key][num_per_group*(i+1):]))
        group_precisions[key].append(sum(precisions[key][num_per_group*(i+1):])/len(precisions[key][num_per_group*(i+1):]))
        group_ncdgs[key].append(sum(ncdgs[key][num_per_group*(i+1):])/len(ncdgs[key][num_per_group*(i+1):]))

    print("\ncur exp setting: {}, {}".format(key,cur_sent_embd_type))
    
    print("Split into {} groups".format(group_num))
    
#    print('\nrecalls:')
#    for r in group_recalls[key]:
#        print(r)
        
    #print('precisions:')#,group_precisions)
    #for p in group_precisions:
    #    print(p)
    
    group_f1s[key]=[]
    for i in range(len(group_precisions[key])):
        if(group_precisions[key][i]+group_recalls[key][i]==0):
            group_f1s[key].append(0)
            print(0)
        else:
            f1=(group_precisions[key][i]*group_recalls[key][i])/(group_precisions[key][i]+group_recalls[key][i])
            group_f1s[key].append(f1)
            
print("top k:",top_k)
print('recalls:',group_recalls)
print('precisions:',group_precisions)
print('f1s:',group_f1s)

#outcome:
#top k: 3
#recalls: {'gpu': [0.6494252873563219, 0.3611111111111111, 0.0], 'ram': [0.9821428571428572, 0.6191374663072776, 0.0], 'hd': [0.8555555555555555, 0.21637426900584797, 0.0], 'screen': [1.0, 0.5, 0.0]}
#precisions: {'gpu': [0.21647509578544058, 0.12037037037037036, 0.0], 'ram': [0.3273809523809524, 0.2063791554357592, 0.0], 'hd': [0.28518518518518515, 0.07212475633528265, 0.0], 'screen': [0.3333333333333333, 0.16666666666666666, 0.0]}
#f1s: {'gpu': [0.16235632183908044, 0.09027777777777778, 0], 'ram': [0.24553571428571427, 0.1547843665768194, 0], 'hd': [0.21388888888888888, 0.05409356725146199, 0], 'screen': [0.25, 0.125, 0]}

#top k: 5
#recalls: {'gpu': [0.7999999999999999, 0.9722222222222222, 0.0], 'ram': [1.0, 1.0, 0.9090909090909092], 'hd': [1.0, 0.6759702286018076, 0.0909090909090909], 'screen': [1.0, 1.0, 1.0]}
#precisions: {'gpu': [0.16, 0.19444444444444442, 0.0], 'ram': [0.2, 0.2, 0.18181818181818182], 'hd': [0.20000000000000004, 0.1351940457203615, 0.01818181818181818], 'screen': [0.2, 0.2, 0.2]}
#f1s: {'gpu': [0.13333333333333333, 0.16203703703703703, 0], 'ram': [0.16666666666666669, 0.16666666666666669, 0.15151515151515152], 'hd': [0.1666666666666667, 0.11266170476696792, 0.015151515151515152], 'screen': [0.16666666666666669, 0.16666666666666669, 0.16666666666666669]}

#for key in top_5_f1:
#    print(key)
#    for each in top_5_f1[key]:
#        print(each)
#    print("\n")
#    
#for key in top_5_recall:
#    print(key)
#    for each in top_5_recall[key]:
#        print(each)
#    print("\n")
#    
#for key in top_3_f1s:
#    print(key)
#    for each in top_3_f1s[key]:
#        print(each)
#    print("\n")
#    
#    
#----------------------new experiment with cpu relabeled outcome:
#
#top k: 5
#recalls: {'gpu': [0.7333333333333334, 0.8888888888888888, 0.0], 'ram': [1.0, 1.0, 0.45454545454545453], 'hd': [1.0, 0.5103668261562998, 0.21212121212121213], 'screen': [1.0, 1.0, 1.0], 'cpu': [0.8916008614501078, 0.33313717902290896, 0.4714285714285714]}
#precisions: {'gpu': [0.1466666666666667, 0.1777777777777778, 0.0], 'ram': [0.2, 0.2, 0.09090909090909091], 'hd': [0.20000000000000004, 0.10207336523125997, 0.04242424242424242], 'screen': [0.2, 0.2, 0.2], 'cpu': [0.17832017229002153, 0.06662743580458179, 0.09428571428571429]}
#f1s: {'gpu': [0.12222222222222225, 0.14814814814814817, 0], 'ram': [0.16666666666666669, 0.16666666666666669, 0.07575757575757576], 'hd': [0.1666666666666667, 0.08506113769271663, 0.03535353535353535], 'screen': [0.16666666666666669, 0.16666666666666669, 0.16666666666666669], 'cpu': [0.14860014357501797, 0.055522863170484826, 0.07857142857142858]}
#
#top k: 3
#recalls: {'gpu': [0.6551724137931035, 0.3549949031600408, 0.0], 'ram': [0.9642857142857143, 0.7247978436657683, 0.0], 'hd': [0.8745298798722693, 0.36921850079744817, 0.0], 'screen': [1.0, 0.5938375350140056, 0.375], 'cpu': [0.6813594916940593, 0.1648441771459814, 0.07142857142857142]}
#precisions: {'gpu': [0.21839080459770113, 0.11833163438668026, 0.0], 'ram': [0.3214285714285714, 0.24159928122192273, 0.0], 'hd': [0.29150995995742307, 0.12307283359914939, 0.0], 'screen': [0.3333333333333333, 0.19794584500466852, 0.125], 'cpu': [0.22711983056468643, 0.054948059048660465, 0.023809523809523808]}
#f1s: {'gpu': [0.16379310344827586, 0.0887487257900102, 0], 'ram': [0.24107142857142858, 0.18119946091644207, 0], 'hd': [0.2186324699680673, 0.09230462519936204, 0], 'screen': [0.25, 0.1484593837535014, 0.09375], 'cpu': [0.17033987292351482, 0.04121104428649535, 0.017857142857142856]}
#