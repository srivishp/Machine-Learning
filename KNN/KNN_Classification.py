# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:48:54 2020

@author: psvis
"""

from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)


#%%    Training and Testing
from sklearn.model_selection import train_test_split
TrainX, TestX, TrainY, TestY = train_test_split(iris.data,iris.target,test_size=0.25)


#%%    Predicting Y Values
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=13,p=2)
knn.fit(TrainX,TrainY)
PredY = knn.predict(TestX)
print("Predicted Values of Y",PredY)


#%%    Printing the Confusion Matrix
from sklearn.metrics import confusion_matrix
confuse = confusion_matrix(TestY,PredY)
print("Confusion Matrix: \n ",confuse)


#%%    Printing Accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(TestY, PredY))
print(metrics.classification_report(TestY, PredY))
print(metrics.confusion_matrix(TestY, PredY))