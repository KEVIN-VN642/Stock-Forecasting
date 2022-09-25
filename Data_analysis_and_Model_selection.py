# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 18:53:02 2022

@author: lenovo
"""
import pandas as pd
import numpy as np
data = pd.read_csv("data\data_clean.csv", index_col= 'Date')

#%%
#Visualize dataset
data.plot.line(subplots = True,figsize = (12,20))
data.hist(figsize=(15,15),bins=25)
# some features looks like Gaussian noises and add no values to prediction
#ft2, ft3, ft4, ft5, ft6, ft7, ft12, ft13, ft14, ft15. They have zero mean and constant variance

#remaining features may add values: ft1, ft8,9,10, 11
#ft8,9,10 look like a truncated normal distribution.

print (data.corr()) #correlation table show zero correlation between Y and noise signals

#%%
# just keep necessary feature
data = data[['Y','ft1','ft11']]
data.to_csv("data\data_for_modelling.csv")
print(data.corr())

#%%
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

plot_acf(data.Y,lags=np.arange(20),title = "Autocorrelation on Y")

plot_pacf(data.Y, lags=np.arange(20), title = "Partial Autocorrelation on Y")

#Above graph show AR(1) model
#Let verify the statement again with acf and pacf plot on differencing Y
plot_acf(data.Y.diff().dropna(),lags=np.arange(20),title = "Autocorrelation on Y.diff()")
plot_pacf(data.Y.diff().dropna(),lags=np.arange(20),title = "Partial Autocorrelation on Y.diff()")
#It do not show signs of AR or MA components so AR(1) should be appropriate model
#%%
pd.plotting.autocorrelation_plot(data.Y.diff().dropna()[-2000:])

#%%
#Check if need to include ft1 and ft11 to model
data = data.assign(Y_shift = data.Y.shift())
data = data.assign(Y_diff = data.Y.diff())
print (data.corr())

#Table shows Y_shift already capture well association between Y and ft1, ft11
#it do not need to include ft1 and ft11


