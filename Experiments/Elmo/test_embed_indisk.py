# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:04:42 2019

@author: Li Xiang

"""

import h5py
import numpy as np

_asin='B07193JRJR'

#filepath='F:\\embedding\\elmo_reviewdata\\'+_asin+'.h5'

ave_filepath='E:\\embedding\\average\\'+_asin+'.h5'
#
#with h5py.File(filepath, 'a') as hf:
#    hf.create_dataset(_asin,  data=np.array([1,2]))
#
with h5py.File(filepath, 'r') as hf:
    read_embeddings=hf[_asin][:]

num_of_sent=read_embeddings.shape[0]
elmo_dimens=1024
total_ave_embding=np.zeros((num_of_sent,elmo_dimens))

#--------get ave----------------------
#j=0
#for each_sent in read_embeddings:
#    word_count=each_sent.shape[0]
#    each_ave_embding=np.zeros((elmo_dimens))
#    for each_word in each_sent:
#        each_ave_embding=each_ave_embding+each_word
#    each_ave_embding=each_ave_embding/word_count
#    total_ave_embding[j]=each_ave_embding
#    j+=1

#--------get max----------------------
j=0
for each_sent in read_embeddings:
    word_count=each_sent.shape[0]
    each_ave_embding=np.zeros((elmo_dimens))
    each_ave_embding=each_sent.max(axis=0)
    total_ave_embding[j]=each_ave_embding
    j+=1

#
#test1=np.zeros(1024)
#test2=np.zeros((num_of_sent,elmo_dimens))
#for rr in read_embeddings[0]:
#    test1=test1+rr
#test1=test1/1914
        
    
    
path='E:\\test_embd.h5'
#with h5py.File(path,'w') as hf:
#    i=0
#    for i in range(len(all_embeddings)):
#        hf.create_dataset(str(i),  data=all_embeddings[i])

new_embd=[]
with h5py.File(path,'r') as hf:
    i=0
    for i in range(47):
        new_embd.append(hf[str(i)][:])
    
    
    
    
    
    
    
    
    
    
    
    
    