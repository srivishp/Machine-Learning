# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:14:30 2020

@author: psvis
"""

import pandas as pd
import numpy as np
#%%
dataset = pd.read_csv('Pima_Indians_Diabetes.csv')
X = dataset.iloc[:,:9].values
Y = dataset.iloc[:,9].values.ravel()


#%%    Predicting missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X)
X = imputer.transform(X)


#%%    Standard Scaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)


#%%    Training and Testing
from sklearn.model_selection import train_test_split
TrainX, TestX, TrainY, TestY = train_test_split(X, Y, test_size = 0.25, random_state = 50)


#%%    Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(TrainX, TrainY)


#%%    Predicting Result
PredY = classifier.predict(TestX)
print("Predicted values of Y",PredY)


#%%    Confusion Matrix
from sklearn.metrics import confusion_matrix
confuse = confusion_matrix(TestY, PredY)
print("Confusion Matrix:",confuse)


#%%    Printing Accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(TestY, PredY))
print(metrics.classification_report(TestY, PredY))
print(metrics.confusion_matrix(TestY, PredY))