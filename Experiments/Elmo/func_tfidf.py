# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:57:25 2019

@author: Li Xiang

some useful functions for tf-idf task
"""

import numpy as np
from math import log

def clean_review(file_list):
    clean_docs=[]
    for review in file_list:
        if(review is None):
            continue
        r=[]
        for w in review.split():
            r.append(w.lower())
        clean_docs.append(r)
    return clean_docs

def get_inverted_index(clean_docs):
    """
    input the file list after cleaning 
    return inverted index
    """
    all_words=[]
    for doc in clean_docs:
        all_words.extend(doc)
    all_words=sorted(set(all_words))
    
    dID_position=[]
    for w in all_words:
        each_word_to_file={}
        for d in range(len(clean_docs)):
            pos_each_doc={}
            if w in clean_docs[d]:
                indices=[i for i,x in enumerate(clean_docs[d]) if x==w]
                dID='D'+str(d+1)
                pos_each_doc={dID:indices}
            if((pos_each_doc)):
                each_word_to_file.update(pos_each_doc)
        if((each_word_to_file)):
            dID_position.append(each_word_to_file)
    return dict(zip(all_words,dID_position))


def get_max_tf(clean_docs):
    """
    input file 
    return max_tf as dictionary type
    """
    max_tf={}
    for d in range(len(clean_docs)):
        max_count=0
        dID='D'+str(d+1)
        c=0
        for w in set(clean_docs[d]):
            c=clean_docs[d].count(w)
            if(c>max_count):
                max_count=c
        max_tf[dID]=max_count
    return max_tf


def get_word_weight(dID,inverted_index,max_tf,each_docs_dic):
    """
    input document ID,  inverted_index, max_tf, and document as dictionary type
    calculate the tf_idf of each word
    return document vecoters as a list of tuples
    """
    N=len(max_tf)
    tf_idf_dic={} 
    weights=[]
    keys=[]
    for w in set(each_docs_dic):
        keys.append(w)
        tf=len(inverted_index[w][dID])
        df=len(list(inverted_index[w].keys()))
        tf_idf_dic[w]=(tf/max_tf[dID])*log(N/df,2)
        weights.append((tf/max_tf[dID])*log(N/df,2))
        
#    word_weight=list(tf_idf_dic.items())
    weight_array=np.array(weights)
    word_weight=list(weight_array/(np.linalg.norm(weight_array)**2))
    word_weight=dict(zip(keys,word_weight))
    
    return word_weight
  