# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:17:42 2023

@author: lorinc
"""
import pandas as pd

#read data frames of 2 strategies
df1=pd.read_csv(r'p:\11208012-011-nabaripoma\Model\Python\results\real\csv\model_output_strategy1.csv')
df2=pd.read_csv(r'p:\11208012-011-nabaripoma\Model\Python\results\real\csv\model_output_strategy2.csv')

print('All CSVs read')

#merge dataframes
df_merged=pd.concat([df1, df2])

print('All CSVs merged')

#write csv
df_merged.to_csv(r'p:\11208012-011-nabaripoma\Model\Python\results\real\csv\model_output_strategies.csv', index=False, float_format='%.2f')

print('Merged CSV saved')