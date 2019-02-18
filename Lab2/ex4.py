#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:59:42 2018

@author: MBP
"""

import matplotlib.pyplot as plt
import matplotlib
import scipy.io as spio
from scipy.stats import multivariate_normal
import numpy as np

def posterior(x,m_mean,m_cov,f_mean,f_cov,pr_m,pr_f):
    cM = pr_m*(np.linalg.det(2*np.pi*m_cov))**(-0.5)
    cM = cM*(np.exp(-0.5*(x-m_mean).dot(np.linalg.inv(m_cov)).dot((x-m_mean).T)))
    cF = pr_f*(np.linalg.det(2*np.pi*f_cov))**(-0.5)
    cF = cF*(np.exp(-0.5*(x-f_mean).dot(np.linalg.inv(f_cov)).dot((x-f_mean).T)))
    res = np.argmax([cM,cF])
    return res
## DATA PREPARATION

mat = spio.loadmat('heightWeight.mat',squeeze_me=True)
raw_data = mat['heightWeightData']
m_X = raw_data[raw_data[:,0]==1,:]
f_X = raw_data[raw_data[:,0]==2,:]

m_train = m_X[0:len(m_X)-25,1:3]
f_train = f_X[0:len(f_X)-35,1:3]
m_test = m_X[len(m_X)-25:,1:3]
f_test = f_X[len(f_X)-35:,1:3]

total_train = len(m_train)+len(f_train)
m_pr = len(m_train)/total_train
f_pr = len(f_train)/total_train

""" TWO-CLASS QUADRATIC DISCRIMINANT ANALYSIS """

m_mean1 = m_train.mean(axis=0).reshape(1,2)
f_mean1 = f_train.mean(axis=0).reshape(1,2)

m_cov1 = np.cov(m_train,rowvar=False)
f_cov1 = np.cov(f_train,rowvar=False)
results_males = [ posterior(x,m_mean1,m_cov1,f_mean1,f_cov1,m_pr,f_pr) for x in m_test]
results_females = [ posterior(x,m_mean1,m_cov1,f_mean1,f_cov1,m_pr,f_pr) for x in f_test]

acc_males = np.sum(np.equal(results_males,0))/len(m_test)
acc_females = np.sum(np.equal(results_females,1))/len(f_test)
accuracy = 100 * (acc_males+acc_females)/2

x1 = range(120,220,1)
x2 = range(30,140,1)
XF1,XF2 = np.meshgrid(x1,x2)
B = np.array([XF1.flatten(),XF2.flatten()])

z =np.asarray( [posterior(x,m_mean1,m_cov1,f_mean1,f_cov1,m_pr,f_pr) for x in B.T])
z = z.reshape(XF1.shape)

plt.figure(1)


plt.contourf(XF1,XF2, z,rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(m_test[:,0],m_test[:,1],c='b')
plt.scatter(f_test[:,0],f_test[:,1],c='r')
plt.title('Two class QDA')
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()



""" TWO-CLASS QUADRATIC DISCRIMINANT ANALYSIS """
#m_mean1 = m_train.mean(axis=0).reshape(1,2)
#f_mean1 = f_train.mean(axis=0).reshape(1,2)

m_cov2 = np.diag(np.diag(m_cov1))
f_cov2 = np.diag(np.diag(f_cov1))

results_males2 = [ posterior(x,m_mean1,m_cov2,f_mean1,f_cov2,m_pr,f_pr) for x in m_test]
results_females2 = [ posterior(x,m_mean1,m_cov2,f_mean1,f_cov2,m_pr,f_pr) for x in f_test]
acc_males2 = np.sum(np.equal(results_males2,0))/len(m_test)
acc_females2 = np.sum(np.equal(results_females2,1))/len(f_test)
accuracy2 = 100*(acc_males2+acc_females2)/2


x1 = range(120,220,1)
x2 = range(30,140,1)
XF1,XF2 = np.meshgrid(x1,x2)
B = np.array([XF1.flatten(),XF2.flatten()])

z =np.asarray( [posterior(x,m_mean1,m_cov2,f_mean1,f_cov2,m_pr,f_pr) for x in B.T])
z = z.reshape(XF1.shape)

plt.figure(2)

plt.contourf(XF1,XF2, z,rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(m_test[:,0],m_test[:,1],c='b')
plt.scatter(f_test[:,0],f_test[:,1],c='r')
plt.title('Two class QDA w diag matrix')
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()



""" LDA Classifier """
t_train = np.append(m_train,f_train,axis=0)
cov = np.cov(t_train,rowvar=False) 
means = (t_train.mean(axis=0)).reshape(1,2)

results_males3 = [ posterior(x,m_mean1,cov,f_mean1,cov,m_pr,f_pr) for x in m_test]
results_females3 = [ posterior(x,m_mean1,cov,f_mean1,cov,m_pr,f_pr) for x in f_test]

acc_males3 = np.sum(np.equal(results_males3,0))/len(m_test)
acc_females3 = np.sum(np.equal(results_females3,1))/len(f_test)
accuracy3 = 100*(acc_males3+acc_females3)/2


x1 = range(120,220,1)
x2 = range(30,140,1)
XF1,XF2 = np.meshgrid(x1,x2)
B = np.array([XF1.flatten(),XF2.flatten()])

z =np.asarray( [posterior(x,m_mean1,cov,f_mean1,cov,m_pr,f_pr) for x in B.T])
z = z.reshape(XF1.shape)

plt.figure(3)

plt.contourf(XF1,XF2, z,rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(m_test[:,0],m_test[:,1],c='b')
plt.scatter(f_test[:,0],f_test[:,1],c='r')
plt.title('Two class LDA')
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()




