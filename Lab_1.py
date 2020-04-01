"""
Created on Wed Feb 19 19:01:46 2020

@author: psvis

"""

import pandas as pd
dataset1 = pd.read_csv("Data1.csv")     #desired number of rows can be printed using "nrows = <no. of rows>"
#print(dataset1)                         #prints entire data set                       
#print(dataset1.iloc[1,:])                #prints specific no. of rows and columns. "iloc[1,:]" prints first row and all columns
######################################################
#%%
x2 = dataset1.iloc[:,4].values
x3 = dataset1.iloc[:,0:4].values
pd.DataFrame(x3)
#%%
from sklearn.preprocessing import Imputer
#predicting the missing values
x1=dataset1.iloc[:,:3].values
impute = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
x1 = impute.fit_transform(x1)
#%%
#Mean
impute = Imputer(strategy = "mean", axis = 0)
meanofx3 = impute.fit_transform(x3[:,0:3])
meanofx2 = impute.fit_transform(x2.reshape(-1,1))
#%%
#Median
impute = Imputer(strategy = "median", axis = 0)
medianofx3 = impute.fit_transform(x3[:,0:3])
medianofx2 = impute.fit_transform(x2.reshape(-1,1))
#%%
#Most Frequent Values
impute = Imputer(strategy = "most_frequent", axis= 0)
freqx3 = impute.fit_transform(x3[:,0:3])
freqx2 = impute.fit_transform(x2.reshape(-1,1))
#%%
#Label Encoding
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
lex3 = x3
lex3[:,3] = label.fit_transform(lex3[:,3])
impute = Imputer(strategy = "mean", axis = 0)
lex3[:,0:3] = impute.fit_transform(lex3[:,0:3])
#%%
#One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
encode = OneHotEncoder(categorical_features = [3])
encx3 = encode.fit_transform(lex3).toarray()
#%%
#One Hot Encoder on the entire dataset
whole = dataset1.iloc[:,:].values
whole[:,3] = label.fit_transform(whole[:,3])
impute = Imputer(strategy = "mean", axis = 0)
whole[:,0:3] = impute.fit_transform(whole[:,0:3])
encode = OneHotEncoder(categorical_features = [3])
whole = encode.fit_transform(whole).toarray()