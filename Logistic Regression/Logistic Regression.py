# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:45:55 2020

@author: psvis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("Kid.csv")
x=dataset.iloc[:,2:18].values
y=dataset.iloc[:,1].values
#%%
#Training and Testing
from sklearn.model_selection import train_test_split
trainx, testx, trainy, testy = train_test_split(x, y, test_size = 0.25, random_state = 0)
#%%
#Calculating the Logistic Regression
from sklearn.linear_model import LogisticRegression
classify = LogisticRegression(random_state = 0)
classify.fit(trainx, trainy) 
predy = classify.predict(testx)
#%%
#Printing the required outputs
from sklearn import metrics
confusion=metrics.confusion_matrix(testy, predy)
accuracy=metrics.accuracy_score(testy, predy)
precision=metrics.precision_score(testy, predy)
recall=metrics.recall_score(testy, predy)
print ("Confusion Matrix : \n",confusion)
print("Accuracy:",'= %.3f' % accuracy)
print("Precision:",'= %.3f' % precision)
print("Recall:",'= %.3f' % recall)
#%%
#Plotting a graph based on class names
class_names=[0,1] 
fig, ax = plt.subplots()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names)
plt.yticks(ticks, class_names)
#%%
# Using seaborn to create a Heatmap
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(testy, predy)), annot=True, cmap="Wistia" ,fmt='g')
ax.xaxis.set_label_position("bottom")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
