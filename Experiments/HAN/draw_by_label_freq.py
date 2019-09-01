# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:08:48 2019

@author: Li Xiang

It's very messy, pay attention when you wanna use it

run draw first then run this file,

remember to change group_num,num array and so on 
"""
import random 
import numpy as np
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
sns.set(style='white')
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

group_num=1
group_map_value=[]
len(label_map_group)

for i in range(math.floor(len(f1_score)/group_num)):
    group_map_value.append(sum(f1_score[group_num*i:group_num*(i+1)])/group_num)

if(len(f1_score[group_num*(i+1):])!=0):
    group_map_value.append(sum(f1_score[group_num*(i+1):])/len(f1_score[group_num*(i+1):]))

#num = np.array([5,10,15,20,23])#for cpu
#num = np.array([2,4,6,8])#for hd
num = np.array([1,2,3,4,5,6])#for ram
sqr = np.array(group_map_value)

d = {'Rank of Labels': num, 'F1 Score': sqr}
pdnumsqr = pd.DataFrame(d)
sns.lineplot(x='Rank of Labels', y='F1 Score', data=pdnumsqr,marker='.')
