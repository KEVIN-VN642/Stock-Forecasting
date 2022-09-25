# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:45:21 2022

@author: Kevin
"""
#%%
import pandas as pd
from Utilities import not_float_number, replace_wrong_values, process_outliers

#reading data
data = pd.read_csv("data/data.csv",index_col="Date")
print (data.shape)
data = data.sort_index()

print (data.info())
#some features are not float type: ft9, ft13, ft14, they are in text format 
#data.ft9[0:3].values array(['0.007195517', '0.010537991', '0.013389128'], dtype=object)

#%%
#Convert string and string-non numeric values to numeric values
positions_ft9, _ = not_float_number(data.ft9)
print("Non numeric position in ft9", positions_ft9)


positions_ft13, _ = not_float_number(data.ft13)
print("Non numeric position in ft13",positions_ft13)


positions_ft14, _ = not_float_number(data.ft14)
print("Non numeric position in ft13",positions_ft14)

# some values is '#?NAME' can not convert to numeric number, need replace them by mean
data.ft9 = replace_wrong_values(data.ft9, positions_ft9)
data.ft13 = replace_wrong_values(data.ft13, positions_ft13)
data.ft14 = replace_wrong_values(data.ft14, positions_ft14)


#%%
data.plot.line(subplots = True,figsize = (12,10))
#We see some clear outlier at ft3, ft5, ft12 and possible ft2

#%%
#replace outliers by mean
data.ft2 = process_outliers(data.ft2)
data.ft3 = process_outliers(data.ft3)
data.ft5 = process_outliers(data.ft5)
data.ft12 = process_outliers(data.ft12)


#%%
#Clean nan values:
print (data.isnull().sum())
#ft1, ft4, ft7, each columns has one null value. We can drop them or do backfill. We choose to backfill here
data.ft1=data.ft1.fillna(method="bfill")
data.ft4=data.ft4.fillna(method="bfill")
data.ft7=data.ft7.fillna(method="bfill")
#check again for nan value
print (data.isnull().sum())

data.to_csv("data\data_clean.csv")










