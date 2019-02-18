#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np

""" KNN CLASSIFIER """
## DATA PREPARATION

mat = spio.loadmat('synthetic.mat', squeeze_me=True)
raw_Train = np.asarray(mat['knnClassify2dTrain'])
raw_Test = np.asarray(mat['knnClassify2dTest'])
print(raw_Test)
plt.figure(1)
cl1 = raw_Test[np.equal(raw_Train[:,2],2)]
cl2 = raw_Test[np.equal(raw_Train[:,2],1)]
plt.scatter(cl1[:,1],cl1[:,0],c='b')
plt.scatter(cl2[:,1],cl2[:,0],c='r')
plt.title('Male & Female Scatter Plot')

def eu_dist(vector1,vector2):
#    res = np.sqrt(np.sum(np.subtract(vector1,vector2)**2))
    res = np.linalg.norm( vector1 - vector2)
    return res

def KNNCL(train,test,k):
    labels=np.asarray([[1,2],[0,0]])
    labels = labels.T
    
    total_train = len(train)
    total_test = len(test)
    
    train_y = train[:,-1] # Train Labels
    train_x = train[:,0:2]

    test_y = test[:,-1] # Test Labels
    test_x = test[:,0:2]
    
    predictions = np.empty([total_test,2])
    
    for test_row in range(0,total_test):
        
        dist = np.empty([total_train,2])
        
        for train_row in range(0,total_train):
            # fill model with distance between test example and all train examples
            dist[train_row,0] = eu_dist(test_x[test_row],train_x[train_row])
            dist[train_row,1] = train_y[train_row]
        
        dist = dist[dist[:,0].argsort()] # sort accessding order
        # Pick first k points
        dist = dist[:k,-1]
        
        for x in range(0,len(labels)):
           labels[x,1] = np.sum(np.equal(dist,labels[x,0]))
        
        predicted_label = labels[np.argmax(labels[:,1],axis=0),0]
        predictions[test_row] = [test_y[test_row],predicted_label]
        
    return predictions

# Plot Accuracy Vs K
acc_list_tst=[]
for k in range(1,50,4):
    mm = KNNCL(raw_Train,raw_Test,k)
    accuracy = (1/len(mm[:,1]))*(np.sum(np.equal(mm[:,0],mm[:,1])))*100
    acc_list_tst.append([accuracy,k])
acc_list_tr=[]
for k in range(1,50,4):
    mm = KNNCL(raw_Train,raw_Train,k)
    accuracy = (1/len(mm[:,1]))*(np.sum(np.equal(mm[:,0],mm[:,1])))*100
    acc_list_tr.append([accuracy,k])
ff= np.asarray(acc_list_tr)
fb = np.asarray(acc_list_tst)
plt.figure(2)
plt.plot(ff[:,1],ff[:,0],fb[:,1],fb[:,0])
plt.show()

