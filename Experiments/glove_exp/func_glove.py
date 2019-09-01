
import h5py
import numpy as np
import os 
import pandas as pd
all_param_list=['cpu','ram','hd','gpu','screen']
all_sent_embd_type=['max','ave','concat','hier']


def get_inds(item, lst):
    """
    get the list according to the min-distance with item
    """
    #sort lst based on difference_lst_
    difference_lst_ = [ abs(x - item) for x in lst ]
    prices = [x for _, x in sorted(zip(difference_lst_, lst))]

    return prices
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

def get_sent_embed(dataset,cur_sent_embd_type,EMBEDDING_DIM,embedding_matrix):
    """
    dataset is a 3-d tensor (datasize, MAX_SENT, MAX_SENT_LEN)
    cur_sent_embd_type in ['max','ave','concat','hier']
    """
    if(cur_sent_embd_type not in all_sent_embd_type):
        print("error!! Wrong input!")
        return
    
    #flat each review and take the non-zero
    #word_index out for computing max
    if(cur_sent_embd_type=='max'):
        data_emb=np.zeros((dataset.shape[0],EMBEDDING_DIM))
        for i in range(dataset.shape[0]):
            nonzero_inds=np.nonzero(dataset[i].flatten())
            review_emb=embedding_matrix[nonzero_inds]
            data_emb[i]=np.amax(review_emb, axis=0)
        
    if(cur_sent_embd_type=='ave'):
        data_emb=np.zeros((dataset.shape[0],EMBEDDING_DIM))
        for i in range(dataset.shape[0]):
            nonzero_inds=np.nonzero(dataset[i].flatten())
            review_emb=embedding_matrix[nonzero_inds]
            data_emb[i]=np.mean(review_emb, axis=0)
            
    if(cur_sent_embd_type=='concat'):
        data_emb=np.zeros((dataset.shape[0],EMBEDDING_DIM*2))
        for i in range(dataset.shape[0]):
            nonzero_inds=np.nonzero(dataset[i].flatten())
            review_emb=embedding_matrix[nonzero_inds]
            review_mean=np.mean(review_emb, axis=0)
            review_max=np.amax(review_emb, axis=0)
            data_emb[i]=np.concatenate([review_mean,review_max])
            
    window_size=3        
    if(cur_sent_embd_type=='hier'):
        data_emb=np.zeros((dataset.shape[0],EMBEDDING_DIM))
        for i in range(dataset.shape[0]):
            word_inds=dataset[i].flatten()[np.nonzero(dataset[i].flatten())]
            
            # if total word count is less than sliding window, we use max
            if(len(word_inds)-window_size<0):
                nonzero_inds=np.nonzero(dataset[i].flatten())
                review_emb=embedding_matrix[nonzero_inds]
                data_emb[i]=np.amax(review_emb, axis=0)
                
            else:# else we use hierachical pooling
                temp_data_emb=np.zeros((len(word_inds)-window_size+1,EMBEDDING_DIM))
                for j,ind in enumerate(word_inds):
                    if(j+window_size>len(word_inds)):
                        break
                    else:
                        words_embed=embedding_matrix[word_inds[j:j+window_size]]
                        temp_data_emb[j]=np.mean(words_embed, axis=0)
                data_emb[i]=np.amax(temp_data_emb, axis=0)
            
    return data_emb
    
def CountFrequency(my_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
      
    for key, value in freq.items(): 
        print ("% d : % d"%(key, value))    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    