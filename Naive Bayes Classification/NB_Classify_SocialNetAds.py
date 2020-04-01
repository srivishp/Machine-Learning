# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:34:30 2020

@author: psvis
"""

import pandas as pd
#%%
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,1:4].values
Y = dataset.iloc[:,4].values


#%%    Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0:1])


#%%    Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


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