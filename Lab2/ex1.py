#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as spio
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd

## DATA PREPARATION

mat = spio.loadmat('heightWeight.mat',squeeze_me=True)
raw_data = mat['heightWeightData']
m_X = raw_data[raw_data[:,0]==1,:]
f_X = raw_data[raw_data[:,0]==2,:] 

plt.figure(1)
plt.scatter(m_X[:,1],m_X[:,2],marker='x',c='b') # Males 
plt.scatter(f_X[:,1],f_X[:,2],marker='o',c='r') # Females
plt.title('Males and Females Data')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

plt.figure(2)
plt.scatter(m_X[:,1],m_X[:,2],marker='x',c='b') # Males 
plt.title('Males Data')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

plt.figure(3)
plt.scatter(f_X[:,1],f_X[:,2],marker='o',c='r') # Females
plt.title('Females Data')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

plt.show()

""" Histograms """
plt.figure(4)
plt.hist(m_X[:,2],11,alpha=0.5,color='b')
plt.hist(f_X[:,2],11,alpha=0.5,color='r')
plt.title('Males & Females Weight Histogram')
plt.xlabel('weight')
plt.ylabel('frequency')
plt.grid(True)
plt.show()

plt.figure(5)
plt.hist(m_X[:,1],11,alpha=0.5,color='b')
plt.hist(f_X[:,1],11,alpha=0.5,color='r')
plt.title('Males & Females Height Histogram')
plt.xlabel('Height')
plt.ylabel('frequency')
plt.grid(True)
plt.show()

""" P(X|y=c,u,E) Maximum Likelihood of mean and Cov"""
#x_m = m_X[:,1:3]

m_mean = np.mean(m_X[:,1:3],axis=0)
f_mean = np.mean(f_X[:,1:3],axis=0)
m_cov = np.cov(m_X[:,1:3],rowvar=False)
f_cov = np.cov(f_X[:,1:3],rowvar=False)
""" MALES """
x1 = range(120,220,1)
x2 = range(30,140,1)
XM1,XM2 = np.meshgrid(x1,x2)
B = np.array([XM1.flatten(),XM2.flatten()])
M = multivariate_normal.pdf(B.T, mean=m_mean, cov=m_cov)
M = M.reshape(len(x2),len(x1))
fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
ax.plot_surface(XM1, XM2, M, color='b',rstride=1, cstride=1, cmap=cm.jet,
linewidth=0, antialiased=False)
plt.title('Males Joing PDF')
plt.show()


""" FEMALES """

x1 = range(120,220,1)
x2 = range(30,140,1)
XF1,XF2 = np.meshgrid(x1,x2)
B = np.array([XF1.flatten(),XF2.flatten()])
F = multivariate_normal.pdf(B.T, mean=f_mean, cov=f_cov)


F = F.reshape(len(x2),len(x1))
fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
ax.plot_surface(XF1, XF2, F, color='b',rstride=1, cstride=1, cmap=cm.jet,
linewidth=0, antialiased=False)
plt.title('Females Joing PDF')
#ax.set_zlim3d(-1.01, 1.01);
plt.show()
plt.figure(9)
plt.figure(1)
plt.scatter(m_X[:,1],m_X[:,2],marker='x',c='b') # Males 
plt.scatter(f_X[:,1],f_X[:,2],marker='o',c='r') # Females
plt.contour(XM1, XM2, M, linewidths=1, colors='k')
plt.contour(XF1, XF2, F, linewidths=1, colors='k')
plt.title('Joint Multivariant Normal PDF')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
#m_cov_MLE = np.sum(x_m.dot(x_m.T))*(1/len(x_m))
#m_cov_MLE -= m_mean.dot(m_mean.T)

#X_train = np.asarray(mat[''])
#y_train = np.asarray(mat['ytrain'])