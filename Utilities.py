# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:45:17 2022

@author: lenovo
"""
import numpy as np

def not_float_number(x):
    positions=[]
    values=[]
    for i in range(len(x)):
        try:
            float(x[i])
        except ValueError:
            positions.append(i)
            values.append(x[i])
    return (positions,values)

def replace_wrong_values(x,positions):
    """
    Change values at positions by mean of other values
    """
    temp=[float(x[i]) for i in range(len(x)) if i not in positions]
    return [float(x[i]) if i not in positions else np.mean(temp) for i in range(len(x))]
     
    
def process_outliers(dat):
    dat_mean, dat_std = np.mean(dat),np.std(dat)
    cut_off=dat_std*3
    lower, upper = dat_mean -cut_off, dat_mean+cut_off

    #calculate mean of normal values:
    normal_mean=np.mean([x for x in dat if x>=lower and x<=upper])
    processed_dat=[x if (x >=lower and x<=upper) else normal_mean for x in dat]
    
    return processed_dat


    
        
    
    
    