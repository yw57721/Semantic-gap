# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:11:48 2019

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

variables for model evaluation:
    
"""

from keras.layers import Input, Dense
from keras.models import Model

#-----------params-----------------------
top_k=5
#-------------------keras multitask model--------------------------------------
label_count={
    'gpu':8,
    'ram':6,
    'hd':11,
    'screen':9,
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

Y_train=[y_train_dic['gpu'],y_train_dic['ram'],y_train_dic['hd'],y_train_dic['screen']]
Y_test=[y_test_dic['gpu'],y_test_dic['ram'],y_test_dic['hd'],y_test_dic['screen']]
Ygen_val=[ygen_val_dic['gpu'],ygen_val_dic['ram'],ygen_val_dic['hd'],ygen_val_dic['screen']]
Ygen_test=[ygen_test_dic['gpu'],ygen_test_dic['ram'],ygen_test_dic['hd'],ygen_test_dic['screen']]

#model.fit(x=X_train,y=Y_train,validation_data=(Xgen_train,Ygen_val),epochs=15)
model.fit(x=X_train,y=Y_train,validation_data=(X_test,Y_test),epochs=15)
print('\n')
model.fit(Xgen_train,Ygen_val,validation_split=0.2,epochs=15)

y_pred_prob=model.predict(Xgen_test)
y_classes={}
for prediction,types in zip(y_pred_prob,list(label_count.keys())):
    y_classes[types] = prediction.argsort()[:,::-1][:,:top_k]

#---------------------------evaluation------------------------------
    
for key in y_classes:
    labels_to_eval=ygen_test_all_label_dic[key][:,:1]
    test_predict_top_k=y_classes[key]
    print("\ncur_exp_param:",key,'\n')
    print("ncdg:")
    i=0
    ndcgs=[]
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



    









