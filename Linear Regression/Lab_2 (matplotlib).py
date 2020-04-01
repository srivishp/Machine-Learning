# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:21:25 2020

@author: psvis
"""

import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Read the data from ".csv" file.
dataset = pd.read_csv('student_scores.csv')
K = dataset[['Hours', 'Scores']]#Taking values for Karl Pearson's Coefficient
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
# First we split the dataset into "Training set" and "Test set"
# Then we split the data in 1/3. Therefore, 20 out of 30 rows will go into "Training set".
# The remaining 10 rows will go into "Test set".
TrainX, TestX, TrainY, TestY = train_test_split(x, y, test_size = 1/3, random_state = 0)
linearRegressor = LinearRegression()
linearRegressor.fit(TrainX, TrainY)
Prediction = linearRegressor.predict(TestX)
plot.scatter(TrainX, TrainY, color = 'magenta')
plot.plot(TrainX, linearRegressor.predict(TrainX), color = 'green')
plot.title('Regression Graph')
plot.xlabel('Hours Studied')
plot.ylabel('Score(%)')
plot.show()
A = K.Hours.corr(K.Scores, method="pearson")
print ("Karl Pearson's Coefficient", '= %.3f' % A) #Karl Pearson's Coefficient upto 3 decimals
