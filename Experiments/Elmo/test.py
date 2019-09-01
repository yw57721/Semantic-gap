# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:16:44 2019

@author: Li Xiang
"""
import pandas as pd
import h5py 
import numpy as np

#df=pd.read_csv("../data/original_dataset.csv",encoding = "ISO-8859-1")
#df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
#df=df[df.reviews.apply(lambda x:len(str(x).split())<=500)]
#
#num_of_sent=0
#for each in all_embeddings:
#    num_of_sent+=each.shape[0]
#    
#    
##num_of_sent=all_embeddings_conct.shape[0]
#elmo_dimens=1024
#total_ave_embding=np.zeros((num_of_sent,elmo_dimens))
#total_max_embding=np.zeros((num_of_sent,elmo_dimens))
#total_concat_embding=np.zeros((num_of_sent,elmo_dimens*2))
#
##----------------1.ave-----------------
#j=0
#for each_batch in all_embeddings:
#    for each_sent in each_batch:
#        word_count=len(x[j].split())
#        each_ave_embding=np.zeros((elmo_dimens))
#        cur_word_ind=0
#        for each_word in each_sent:
#            if(cur_word_ind>=word_count):
#                break
#            each_ave_embding=each_ave_embding+each_word
#            cur_word_ind+=1
#        each_ave_embding=each_ave_embding/word_count
#        total_ave_embding[j]=each_ave_embding
#        j+=1
#    
##----------------2.max-----------------
#j=0
#for each_batch in all_embeddings:
#    for each_sent in each_batch:
#        word_count=len(x[j].split())
#        each_max_embding=np.zeros((elmo_dimens))
#        each_max_embding=each_sent[:word_count,:].max(axis=0)
#        total_max_embding[j]=each_max_embding
#        j+=1
#
#
##----------------3.concantenate-----------------
ave_filepath='E:\\embedding_updated\\elmo_reviewdata\\average\\'+_asin+'.h5'
max_filepath='E:\\embedding_updated\\elmo_reviewdata\\max\\'+_asin+'.h5'
con_filepath='E:\\embedding_updated\\elmo_reviewdata\\concat\\'+_asin+'.h5'
    
#total_concat_embding=np.concatenate((total_ave_embding,total_max_embding),axis=1)
with h5py.File(ave_filepath, 'w-') as hf:
    hf.create_dataset(_asin,  data=total_ave_embding)
#-----write max embedding file------------------
with h5py.File(max_filepath, 'w-') as hf:
    hf.create_dataset(_asin,  data=total_max_embding)
#-----write con embedding file------------------
with h5py.File(con_filepath, 'w-') as hf:
    hf.create_dataset(_asin,  data=total_concat_embding)
  
print('{}: {}/{} finished!'.format(_asin,num,asin_length))