# -*- coding: utf-8 -*-
"""

WARNNING:: don't run this program!!

Created on Tue Mar 12 14:32:26 2019

@author: Li Xiang

1.get the computer reviews' elmo embedings by each asin number
2.write to disk(via h5 to filpath)
3.each h5 file is named by asin number and dataset name is also asin

"""

import sys
import h5py
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd 
import numpy as np

#sys.exit(0)
df=pd.read_csv("../data/original_dataset.csv",encoding = "ISO-8859-1")
df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
df=df[df.reviews.apply(lambda x:len(str(x).split())<=500)]

df=df[df.asin.str.startswith('B')]
#df=pd.read_csv("../data/generated_reviews.csv")


all_asins=list(df['asin'].unique())

asin_review=dict()

for _asin in all_asins: 
    asin_review[_asin]=df[df.asin==_asin]['reviews'].tolist()

#--------def functions and get max review length-----------------------------------------------------

max_review_length = 0 # in this example max_review_length==499
i=0
flag=0
for r in df.reviews.tolist():
    i+=1
    if((max_review_length<len(r.split()))):
        flag=i
        max_review_length=max(max_review_length,len(r.split()))
        rr=r

def pad(e, sentence_length=max_review_length):
    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    num_sentences, old_sentence_length, embedding_length = e.shape
    e2 = np.zeros((num_sentences, sentence_length, embedding_length))
    e2[:, :old_sentence_length , :] = e
    return e2


elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


asin_length=len(asin_review.keys())

startflag=0#37

num=startflag
batch_step = 16
#sys.exit()
for _asin in list(asin_review.keys())[startflag:]:
    num+=1
    all_embeddings=[]
    x=asin_review[_asin]
    
    if(len(x)<=batch_step):
        embeddings = elmo_model(
                x,
                signature="default",
                as_dict=True
            )["elmo"]
        e = sess.run(embeddings)
        e = pad(e)
        all_embeddings.append(e)
        
    if(len(x)>batch_step):
        for i in range(int(len(x)/batch_step)+1):
            print('In num {} : {}/{}'.format(num,i+1,int(len(x)/batch_step)+1))
            left = i*batch_step
            right = (i+1)*batch_step
            this_x = x[left:right]
        
            # due to the +1 in the range(...+1), we can end up
            # with an empty row at the end. just skip it.
            if not this_x:
                continue
        
            embeddings = elmo_model(
                this_x,
                signature="default",
                as_dict=True
            )["elmo"]
            e = sess.run(embeddings)
            e = pad(e)
            all_embeddings.append(e)
            
    #all_embeddings_conct is the final sent embedding
    all_embeddings_conct = np.concatenate(all_embeddings)
    print(all_embeddings_conct.shape) 
    
#    filepath='D:\\Datasets\\elmo\\elmo_generate_reviewdata\\'+_asin+'.h5'
    filepath='D:\\Datasets\\elmo\\elmo_train_reviewdata\\'+_asin+'.h5'
    
    with h5py.File(filepath, 'w-') as hf:
        hf.create_dataset(_asin,  data=all_embeddings_conct)

    print('{}: {}/{} finished!'.format(_asin,num,asin_length))


sess.close()