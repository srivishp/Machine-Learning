# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:52:43 2020

@author: akhil
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("Social_Network_Ads (2).csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clsfier=DecisionTreeClassifier(random_state=0)
clsfier.fit(X_train,y_train)

tree.plot_tree(clsfier)
#tree.export_graphviz(clsfier,out_file='tree.pdf')
