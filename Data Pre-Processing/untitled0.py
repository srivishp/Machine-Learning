# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:01:46 2020

@author: psvis
"""

import pandas as pd
dataset = pd.read_csv("Data1.csv")
#%%
allrow = dataset.iloc[:,:] #displaying the full data
x = dataset.iloc[0:11,:]   #displaying for 10 rows
#%%
dep = dataset.iloc[:,4].values #profit #dependent data(y)
indep = dataset.iloc[:,0:4].values 
pd.DataFrame(indep)
#rest  #independent data(x) (-1 to remove last column) 
#%%
#to fill missing values
from sklearn.preprocessing import Imputer
#%%
#mean #filling with mean
impute = Imputer(strategy = "mean", axis = 0)#object for imputer
meanindep = impute.fit_transform(indep[:,0:3])
meandep = impute.fit_transform(dep.reshape(-1,1))
#%%
#median #filling with median
impute = Imputer(strategy = "median", axis = 0)
medindep = impute.fit_transform(indep[:,0:3])
meddep = impute.fit_transform(dep.reshape(-1,1))
#%%
#mostfrequent #filling with mode
impute = Imputer(strategy = "most_frequent", axis= 0)
mosindep = impute.fit_transform(indep[:,0:3])
mosdep = impute.fit_transform(dep.reshape(-1,1))
#%%
#labelencoding
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
leindep = indep
leindep[:,3] = label.fit_transform(leindep[:,3])
impute = Imputer(strategy = "mean", axis = 0)
leindep[:,0:3] = impute.fit_transform(indep[:,0:3])
#%%
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [3])
oeindep = enc.fit_transform(leindep).toarray()
#%%
#on whole dataset
wdata = dataset.iloc[:,:].values
wdata[:,3] = label.fit_transform(wdata[:,3])
impute = Imputer(strategy = "mean", axis = 0)
wdata[:,0:3] = impute.fit_transform(wdata[:,0:3])
enc = OneHotEncoder(categorical_features = [3])
wdata = enc.fit_transform(wdata).toarray()