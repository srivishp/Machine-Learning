# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:55:32 2020

@author: psvis
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr

def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    xmean, ymean = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*ymean*xmean 
    SS_xx = np.sum(x*x) - n*xmean*xmean 
  
    # calculating regression coefficients 
    b1 = SS_xy / SS_xx 
    b0 = ymean - b1*xmean 
  
    return(b0, b1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 
  
def main(): 
    # observations 
    x = np.array([50,63,70,86,90,107,110,128,132,150]) 
    y = np.array([35,53,67,89,95,107,119,125,134,146]) 
    
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb0 = {}  b1 = {}".format(b[0], b[1])) 
    
  
    # calculate Karl Pearson's coefficient
    A, _ = pearsonr(x, y)
    print("Karl Pearson's Coefficient", '= %.3f' % A)
    
    # plotting regression line 
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main()