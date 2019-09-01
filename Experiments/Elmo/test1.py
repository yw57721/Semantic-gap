# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:03:01 2019

@author: Li Xiang

correct the results for average elmo embedding

"""
import pandas as pd
import h5py 
import numpy as np


df=pd.read_csv("../data/original_dataset.csv",encoding = "ISO-8859-1")
df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)

#delete some random noise in dataset
df=df[df.asin.str.startswith('B')]

all_asins=list(df['asin'].unique())

max_review_length = 0
i=0
flag=0
for r in df.reviews.tolist():
    i+=1
    if((max_review_length<len(r.split()))):
        flag=i
        max_review_length=max(max_review_length,len(r.split()))
        rr=r
        
asin_review=dict()

for _asin in all_asins: 
    asin_review[_asin]=df[df.asin==_asin]['reviews'].tolist()


#start editing
i=0
for _asin in all_asins:
    ave_path='..//data//elmo_reviewdata//average//'+_asin+'.h5'
    with h5py.File(ave_path,'r') as hf:
        cur_ave_embd=hf[_asin][:]
    max_path='..//data//elmo_reviewdata//max//'+_asin+'.h5'
    with h5py.File(max_path,'r') as hf:
        cur_max_embd=hf[_asin][:]
        
    cur_concat_embd=np.concatenate((cur_max_embd,cur_ave_embd),axis=1)    
    con_filepath='..\\data\\elmo_reviewdata\\concat\\'+_asin+'.h5'
#    con_filepath='E:\\embedding\\elmo_reviewdata\\new_concat\\'+_asin+'.h5'
    with h5py.File(con_filepath,'w-') as hf:
        hf.create_dataset(_asin,  data=cur_concat_embd)
    i+=1    
    print("{}/112 finished!".format(i))
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
