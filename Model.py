# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:31:43 2022

@author: lenovo
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class stockprice_fore():
    def __init__(self, lags):
        """
        Initialize attributes of class
        - X: input time series, updated when start training
        - coef: keeping coeficients of linear regression model
        - intercept: keeping intercept value of linear regresion model
        - lags: identify the order of AR(p) 
        """
        self.X = None
        self.coef = None
        self.intercept = None
        self.lags = lags
        self.preds = None

    def fit(self, X):
        """
        training method
        - X: input time series
        """
        self.X = pd.DataFrame(X)
        self.X = self.X.rename(columns={self.X.columns[0] : 'Y'})
        for i in range(1,self.lags +1):
            self.X['Y'+str(i)] = self.X.Y.shift(i)
        self.X = self.X.dropna()
        Reg = LinearRegression()
        Reg.fit(self.X.drop(columns=['Y']), self.X.Y)
        self.coef = Reg.coef_
        self.intercept = Reg.intercept_
    
    def forecast(self,k):
        """
        predict/forecast method
        k: number of values to forecast ahead
        """
        if k<=0:
            raise Exception("K must be a positive integer")
        
        n = self.X.shape[0]
        pred = [0] * k
        pred[0] = sum(self.coef * self.X.Y[(n-self.lags):n])  + self.intercept
        
        if k <= self.lags:
            for i in range(1,k):
                pred[i] = sum(self.coef[0:(self.lags - i)] * self.X.Y[(n-self.lags+i):n]) + \
                    sum(self.coef[(self.lags - i): (self.lags)] * pred[0:i]) + self.intercept
            
        
        if k > self.lags:
            for i in range(1,self.lags):
                pred[i] = sum(self.coef[0:(self.lags - i)] * self.X.Y[(n-self.lags+i):n]) + \
                    sum(self.coef[(self.lags - i): (self.lags)] * pred[0:i]) + self.intercept
            
            for i in range(self.lags,k):
                pred[i] = sum(self.coef * pred[(i-self.lags):i]) + self.intercept
        self.preds = pred
                
        return pred
    
    def plot_forecast(self,test=None ,l=1000):
        """
        this method plot a part of training data (measure by l) and 
        test data(if provided, used when we want validate performance of model)
        with forecasting values
        test: a pandas series, used to compare with forecasting values
        l: number of observations in training dataset to display
        
        """
        frame = min(len(self.X),l)
      
        if (test is not None):
            # This case is used to test model and we known test data in advance
            train = self.X['Y']
            test.index.name = 'Date'
            
            train = train.to_frame()
            train = train.reset_index()

            test = test.to_frame()
            test = test.reset_index()
            test['pred'] = self.preds

            combine = pd.concat([train,test])

            combine = combine.merge(train, on='Date', how='left')

            combine = combine.merge(test, on='Date', how='left')

            combine.columns = ['Date', 'All', 'Pred_1', 'Train', 'Test','Pred']
            combine = combine.drop(columns=['All','Pred_1'])
            
            combine = combine[(len(train) - frame + len(test)):]
            combine = combine.set_index('Date')
            combine.plot(figsize = (12,5), title='Validation Stock Price')
            
        if (test is None):
            #Test is not provided, imply that this data is not availble at the time
            #of forecasting, since the trading days are unknown (due to holidays...)
            #we do not display date, only order of data is shown
            train = self.X['Y']
            train_dp = train[len(train)-frame:]
            
            df = pd.DataFrame({"Known observations": list(train_dp.values)+[np.nan]*len(self.preds),
                              "Forecast": [np.nan]*len(train_dp.values) +self.preds
                              })
            df.plot(figsize = (12,5), title='Stock Price Forecasting')

    
    
    