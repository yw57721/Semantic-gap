# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:53:37 2019

@author: Li Xiang
"""
import h5py
import numpy as np
import os 
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

def get_new_cpu_useful_asin():
    asin_lst=[]

    fpath='..//data//'+'asin_map_'+'ram'+'_rank.npy'    
    asin_map_labels=np.load(fpath).item()
    for asin in list(asin_map_labels.keys()):
        asin_lst.append(asin)
        
    return asin_lst

def get_new_cpu_label():
    """
    give cpu new label 0-9, in order to solve the problem of missing values
    
    cpu label mapping table:
        Description	                Label
        Intel Celeron/ADM A (0, 2GHz)	0
        Intel Celeron/ADM A ([2, 3)GHz	1
        Intel Celeron/ADM A ([3, )GHz	2
        Intel i3 (0, 2.4) GHz	         3
        Intel i3 [2.4, ) GHz	         4
        Intel i5 (0, 2] GHz	         5
        Intel i5 (2, 3) GHz	         6
        Intel i5 [3, ) GHz	            7
        Intel i7 (0, 2] GHz	         6
        Intel i7 (2, 3] GHz	         7
        Intel i7 [3, ) GHz	            8
        Others	                        9    
    """
    
    cpu_tech_file='..//data//amazon_tech_cpus_1207.json'
    asin_map_labels = {}
    with open(cpu_tech_file, 'r') as f1:
        for line in f1:
#            if '+' in line:
                #ind += 1
            asin = line.split(':')[0].strip()
            cpu_model = line.split(':')[1].strip()
            md_lst=cpu_model.split()
            Ghz=0
            #get the GHz number
            if(len(md_lst)>1 and md_lst[1]=='GHz'):
                Ghz=float(md_lst[0])
            if(('Celeron' in cpu_model)or('AMD' in cpu_model)):
                if(Ghz<2):
                    label=[0]
                elif(Ghz<3):
                    label=[1]
                else:
                    label=[2]
            elif('i3' in cpu_model):
                if(Ghz<2.4):
                    label=[3]
                else:
                    label=[4]
            elif('i5' in cpu_model):
                if(Ghz<2):
                    label=[5]
                elif(Ghz<3):
                    label=[6]
                else:
                    label=[7]
            elif('i7' in cpu_model):
                if(Ghz<2):
                    label=[6]
                elif(Ghz<3):
                    label=[7]
                else:
                    label=[8]
            else:
                label=[9]
                
            asin_map_labels[asin]=label
                        
    return asin_map_labels          
    
                    
                

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
                ind_lst.append(price_map_label[price])
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
                    
def get_review_length(asin_review_dic,asin):
    sent_len=[]
    for sent in asin_review_dic[asin]:
        sent_len.append(len(sent.split()))
    return sent_len

def get_gener_review_embedding(emb_type,asin):
    
    fpath='..//data//elmo_generate_reviewdata//'
    
    if((emb_type=='max')):
        sub_path='max//'+asin+'.h5'
        file=fpath+sub_path
        if(os.path.exists(file)==False):
            return None
        else:
            with h5py.File(file,'r') as hf:
                cur_sent_embd=hf[asin][:]
        
    if((emb_type=='ave')):
        sub_path='average//'+asin+'.h5'
        file=fpath+sub_path
        if(os.path.exists(file)==False):
            return None
        else:
            with h5py.File(file,'r') as hf:
                cur_sent_embd=hf[asin][:]
        
    if((emb_type=='concat')):
        sub_path='concat//'+asin+'.h5'
        file=fpath+sub_path
        if(os.path.exists(file)==False):
            return None
        else:
            with h5py.File(file,'r') as hf:
                cur_sent_embd=hf[asin][:]
    if(emb_type not in all_sent_embd_type):
        print('error: func get_sent_embedding input error!! ')
    
    return cur_sent_embd    

def get_sent_embedding(emb_type,asin):
    """
    return current embeddings in a numpy array
    based on embedding type and asin number
    
    emb_type in ['max','ave','concat']
    """
    fpath='..//data//elmo_reviewdata//'
    
    if((emb_type=='max')):
        sub_path='max//'+asin+'.h5'
        file=fpath+sub_path
        with h5py.File(file,'r') as hf:
            cur_sent_embd=hf[asin][:]
        
    if((emb_type=='ave')):
        sub_path='average//'+asin+'.h5'
        file=fpath+sub_path
        with h5py.File(file,'r') as hf:
            cur_sent_embd=hf[asin][:]
        
    if((emb_type=='concat')):
        sub_path='concat//'+asin+'.h5'
        file=fpath+sub_path
        with h5py.File(file,'r') as hf:
            cur_sent_embd=hf[asin][:]
    if(emb_type not in all_sent_embd_type):
        print('error: func get_sent_embedding input error!! ')
    
    return cur_sent_embd    


if __name__ == "__main__":
    pass
#    test=get_sent_embedding('concat','B06XFGF7SN')
#    test=get_sorted_output_list('cpu')
    