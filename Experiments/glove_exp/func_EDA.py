# -*- coding: utf-8 -*-
"""

4 augmentation method from paper 
<EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classiﬁcation Tasks>

Created on Sun May 26 18:19:45 2019

@author: Li Xiang

"""


import random
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

stopword=set(stopwords.words('english'))
#sent='You can even modify the list by adding words of your choice in the english .txt. file in the stopwords directory.'

##--------------Synonym Replacement (SR)------------------
#
#--1.Synonym Replacement (SR): Randomly
#--choose n words from the sentence that are not
#--stop words. Replace each of these words with
#--one of its synonyms chosen at random.

def synonym_replacement(alpha,sent):
    """
    input: alpha(percentage of words in a sentence proceed EDA),sent (original sentence)
    
    output: sent (after Synonym Replacement)
        
    """
#    alpha=0.2
    word_seq=text_to_word_sequence(sent)
    n_replace=int(len(word_seq)*alpha)
    
    replace_indices=[]
    
    sent_no_stopw=[w for w in word_seq if w not in stopword]
    
    break_flag=0
    while(len(replace_indices)<n_replace):
        if(len(sent_no_stopw)<1):
            break
        rd=random.randint(0,len(sent_no_stopw)-1)
        if((rd not in replace_indices)and(wordnet.synsets(sent_no_stopw[rd]))):
            replace_indices.append(rd)
        break_flag+=1
        if(break_flag>50):
            break
#    print('end while')    
    replace_words=[sent_no_stopw[i] for i in replace_indices]
    
    synonyms = []
    replace_map={}
    for word in replace_words:
        for syn in wordnet.synsets(word):
        		for l in syn.lemmas():
        			synonyms.append(l.name())
        synonyms=list(set(synonyms))
        replace_map[word]=random.choice(synonyms)
#    print(replace_map)
    
    # a little problem here, because split may contain punctuations with words
    sent_new=sent.split()
    for ind,w in enumerate(sent_new):
        if(w in replace_map):
            sent_new[ind] = replace_map[w]
    sent_new=' '.join(sent_new)
    
    sent=sent_new
    return sent

#-----------------2. Random Insertion (RI): 

#Find a random synonym of a random word in the sentence that is
#not a stop word. Insert that synonym into a ran-
#dom position in the sentence. Do this n times.


def random_insertion(alpha,sent):
#    alpha=0.5
    word_seq=text_to_word_sequence(sent)
    n_insert=int(len(word_seq)*alpha)
    
    word_seq=text_to_word_sequence(sent)    
    sent_no_stopw=[w for w in word_seq if w not in stopword]
    if(sent_no_stopw==[]):
        return sent
    random_word=random.choice(sent_no_stopw)
    j=0
    while(wordnet.synsets(random_word)==[]):
        random_word=random.choice(sent_no_stopw)
        if(j>50):
            return sent
        j+=1
    
    synonyms=[]
    for syn in wordnet.synsets(random_word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    synonyms=list(set(synonyms))  
    
    #insertion
    sent_list=sent.split()
    for i in range(n_insert):
        rand_syn=random.choice(synonyms)
        ind=random.randint(0,len(sent_list)-1)
        sent_list.insert(ind,rand_syn)
        
    new_sent=' '.join(sent_list)
#    if(sent!=new_sent):
#        print('yes')
    return new_sent
    
#-----------------3. Random Swap (RS): 
#Randomly choose two words in the sentence and swap their positions.
#Do this n times.
    
#sent='You can even modify the list by adding words of your choice in the english .txt. file in the stopwords directory.'

def random_swap(alpha,sent):
    word_seq=text_to_word_sequence(sent)
    n_swap=int(len(word_seq)*alpha)
#    print(n_swap)
    sent_list=sent.split()
    
    for i in range(n_swap):
        ind1=random.randint(0,len(sent_list)-1)
        ind2=random.randint(0,len(sent_list)-1)
        exch_w=sent_list[ind1]
        sent_list[ind1]=sent_list[ind2]
        sent_list[ind2]=exch_w
    
    sent=' '.join(sent_list)
    return sent

#-----------------4. Random Deletion (RD): Randomly remove
#each word in the sentence with probability p.

def random_deletion(prob,sent):
    
    sent_list=sent.split()
    
    for i,w in enumerate(sent_list):
        rd=random.random()
        if(rd<prob):
            del sent_list[i]
    sent=' '.join(sent_list)
    return sent
    











