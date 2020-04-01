# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:53:04 2020

@author: psvis
"""

import pandas as pd
dataset = pd.read_csv("Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(x, y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
classify=DecisionTreeClassifier(random_state=0)
classify.fit(Train_X,Train_Y)

tree.plot_tree(classify)
tree.export_graphviz(classify,out_file='tree.txt')