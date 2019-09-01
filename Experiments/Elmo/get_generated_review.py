# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:58:31 2019

@author: Li Xiang

load customer generated review in 2 data file and write to 1 file : generated_reviews.csv

"""

import pandas as pd
import numpy as np

def get_repharased_asins(sheets):
    asins = {}
    for name, sheet in sheets.items():
        rephrased_col = sheet.keys()[-1]
        reviews = sheet[rephrased_col]
        review_lst = reviews[reviews.notnull()].reset_index(drop=True).tolist()
        if(review_lst!=[]):
            asins[name]=review_lst        
    return asins

file0 = '..//data//amazon_0.xlsx'
file1 = '..//data//amazon_1.xlsx'

sheets_0 = pd.read_excel(file0, sheet_name=None)
sheets_1 = pd.read_excel(file1, sheet_name=None)

asins_0 = get_repharased_asins(sheets_0)
asins_1 = get_repharased_asins(sheets_1)

generated_asins={}
generated_asins.update(asins_0)
generated_asins.update(asins_1)

new_list=[]
for asin,reviews in generated_asins.items():
    for rev in reviews:
        new_list.append([asin,rev])

generate_rev_df=pd.DataFrame(data=new_list,columns=['asin','reviews'])
out_filepath='..//data//generated_reviews.csv'
generate_rev_df.to_csv(out_filepath,index=False)

del asins_0
del asins_1
del sheets_0
del sheets_1

