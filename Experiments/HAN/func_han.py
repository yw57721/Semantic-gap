# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:53:37 2019

@author: Li Xiang
"""
import h5py
import numpy as np
import os 
import pandas as pd
all_param_list=['cpu','ram','hd','gpu','screen']
all_sent_embd_type=['max','ave','concat']

def get_inds(item, lst):
    """
    get the list according to the min-distance with item
    """
    #sort lst based on difference_lst_
    difference_lst_ = [ abs(x - item) for x in lst ]
    prices = [x for _, x in sorted(zip(difference_lst_, lst))]

    return prices
#def new_get_sorted_label_list(exp_param):
#    asin_map_labels={}
#    return 

def get_sorted_label_list(exp_param):
    """
    input:exp_param
    output: a dictionary {asin:list of sorted params}
    
    """
    
    if(exp_param not in all_param_list):
        print("error:func get_sorted_output_list input not defined!!!")
        return 0
    
    if(exp_param=='cpu'):
        
        asin_lst=[]
        price_lst=[]
        label_lst=[]
        asin_map_price={}
        price_map_label={}
        cpu_tech_file='..//data//amazon_tech_cpus_1207.json'
    
        with open(cpu_tech_file, 'r') as f1:
            for line in f1:
                if '+' in line:
                    #ind += 1
                    asin = line.split(':')[0].strip()
                    asin_lst.append(asin)
                    price = int(line.split(':')[3].strip())
                    _label = int(line.split(':')[2].strip())
                    
                    asin_map_price[asin] = price # the duplication
                    price_map_label[price] = _label
                    price_lst.append(price)
#                    label_lst.append(_label)
        asin_map_labels = {}     
        for asin in asin_map_price:
            price = asin_map_price[asin]
            prices_lst = get_inds(price, price_lst)
            ind_lst = []
            for price in prices_lst:
                #new add line
                if(price_map_label[price] not in ind_lst):
                    ind_lst.append(price_map_label[price])
#            print(ind_lst)
            
            #each asin and its ind_lst(ind_lst is sorted by price distance)
            asin_map_labels[asin] = ind_lst
                    
        return asin_map_labels
    
    if(exp_param in ['ram','gpu','hd','screen']):
        fpath='..//data//'+'asin_map_'+exp_param+'_rank.npy'    
        asin_map_labels=np.load(fpath).item()
        return asin_map_labels
    
    return

def get_useful_asins(exp_param):
    """
    return the asin numbers as a list
    
    in each experiment, the useful asins may be different
    here we get the needed ones based on each file 
    
    exp_param in ['cpu','ram','hd','gpu','screen']
    """
    if(exp_param not in all_param_list):
        print("error:func get_useful_asins input not defined!!!")
        return 0    
    
    asin_lst=[]
    if(exp_param=='cpu'):
        cpu_tech_file='..//data//amazon_tech_cpus_1207.json'
        with open(cpu_tech_file, 'r') as f1:
            for line in f1:
                if '+' in line:
                    #ind += 1
                    asin = line.split(':')[0].strip()
                    asin_lst.append(asin)
                    
    if(exp_param in ['ram','gpu','hd','screen']):
        fpath='..//data//'+'asin_map_'+exp_param+'_rank.npy'    
        asin_map_labels=np.load(fpath).item()
        for asin in list(asin_map_labels.keys()):
            asin_lst.append(asin)
            
    if(asin_lst==[]):
        print('error: asin_lst not obtained!!')
    
    return asin_lst
                    
        
def get_gener_review(asin_list):
    gene_rev_file='..//data//generated_reviews.csv'
    df_val=pd.read_csv(gene_rev_file)
    reviews=df_val.reviews.tolist()
    review_dic={}
    
    for _asin in asin_list:
        index_list=df_val.index[df_val['asin'] == _asin].tolist()
        if(index_list!=[]):
            review_dic[_asin]=[reviews[i] for i in index_list]
    
    return review_dic    

def get_reviews(reviews,asin):
    """
    return reviews dic
    based on df and asin number
    
    """
    
    rev_dic={}
    
    for each_asin in asin:
        index_list=df.index[df['asin'] == each_asin].tolist()
        rev_dic[each_asin]=[reviews[i] for i in index_list]
    
    
    return rev_dic    


if __name__ == "__main__":
    pass
#    test=get_sent_embedding('concat','B06XFGF7SN')
#    test=get_sorted_output_list('cpu')
    