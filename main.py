import numpy as np
import pandas as pd
import os
#os.chdir("C:\\Users\\lenovo\\Desktop\\Alphalayer")
# Plots
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

# Modeling and Forecasting
# ==============================================================================
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from Model import stockprice_fore

#%% reading data
data= pd.read_csv("data/data_for_modelling.csv",index_col='Date')

#%% 
Y = data.Y

#We do not need to use too long historical data, we keep only data from 2006 upwards
Y = Y[Y.index >= '2000-01-01'] # use 22 years back data



#%% Validating the model

steps = 80
train = Y[-2500-steps:-steps]
test = Y[-steps:]
ar= stockprice_fore(lags=1)
ar.fit(train)
preds = ar.forecast(steps)
print ("RMSE :", round(np.sqrt(mean_squared_error(test,preds)),2))
print ("R2: ", round(r2_score(test, preds),2))
ar.plot_forecast(test, l=500,)


#%% Forecast future values (this case test set is unknow)
ar= stockprice_fore(lags=1)
ar.fit(Y[-2500:])
preds = ar.forecast(steps)
ar.plot_forecast(l=500)

#%%
#####################################################################################
############EXPANSION################EXPANSION################EXPANSION##############






