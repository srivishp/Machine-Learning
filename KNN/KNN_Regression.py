# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:37:41 2020

@author: psvis
"""
import pandas as pd
from sklearn import neighbors
import numpy as np


#%%    Importing the dataset
dataset = pd.read_csv('Automobile_data.csv')
dataset.replace('?',np.NaN,inplace=True)
X = dataset.iloc[:,1:].values
Y = dataset.iloc[:,0].values


#%%    Predicting the missing values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values = np.NaN, strategy = 'most_frequent',verbose=0)
impute = impute.fit(X)
X = impute.transform(X)


#%%    Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(1,8):
 X[:,i] = le.fit_transform(X[:,i])
X[:,13] = le.fit_transform(X[:,13])
X[:,14] = le.fit_transform(X[:,14])
X[:,16] = le.fit_transform(X[:,16])


#%%    Training and Testing
from sklearn.model_selection import train_test_split
TrainX, TestX, TrainY, TestY = train_test_split(X,Y,test_size=0.25,random_state=0)


#%%    Regression
n_neigbors = 13
knn = neighbors.KNeighborsRegressor(n_neigbors)


#%%
knn.fit(TrainX,TrainY)

#%%
PredY = knn.predict(TestX)


#%%    Pearson Correlation
print(TestY)
print(PredY)
correlate = dataset.corr(method='pearson')
print(correlate)