#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:05:00 2018

@author: MBP
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as spio
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd

## DATA PREPARATION
mat = spio.loadmat('Indian_Pines_Dataset.mat',squeeze_me=True)
indian_pines = mat['indian_pines']
indian_pines_gt = mat['indian_pines_gt']

size_class1 = 1265
size_class2 = 1428

class1 = np.zeros([size_class1,220])
class2 = np.zeros([size_class2,220])

n1=0
n2=0

for i in range(0,indian_pines.shape[0]):
    for j in range(0,indian_pines.shape[1]):
        if indian_pines_gt[i,j]==14:    
            class1[n1,:] = indian_pines[i,j,:]
            n1+=1
        if indian_pines_gt[i,j]==2:
            class2[n2,:] = indian_pines[i,j,:]
            n2+=1

class1n = class1 - class1.mean(axis=0)

class1n = np.devide(class1n,class1n.std(axis=0))

class2n= class2-class2.mean(axis=0)
class2 = class2n/class2n.std(axis=0)

train_data = np.append(class1n,class2n,axis=0)

train_mean = train_data.mean(axis=0)

#train_data -= train_mean

train_cov = np.cov(train_data,rowvar=False)

w,v = np.linalg.eigh(train_cov)
K = 150 # number of prinicipal components 
W=np.flip(v,axis=1)
W=W[:,:K]
Z = np.zeros([train_data.shape[0],K]) # vector holding the predicted values
#Z = (W.T).dot(train_data)

for i in range(0,Z.shape[0]):
    Z[i] = (W.T).dot(train_data[i])

# x-WZ
""" CALCULATE MSE """ 
mse = 0 

for i in range(0,train_data.shape[0]):
    x_hat = W.dot(Z[i])
    mse+= (train_data[i] - x_hat).dot((train_data[i]-x_hat).T)
    
mse= mse/train_data.shape[0]


#raw_data = mat['heightWeightData']
#m_X = raw_data[raw_data[:,0]==1,:]
#f_X = raw_data[raw_data[:,0]==2,:] 