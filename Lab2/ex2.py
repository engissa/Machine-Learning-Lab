
# coding: utf-8

# # Exercise 2 
# ### Model fitting for discrete distributions: Bag of Words
# 


"""
@author: Mohammad Issa 
"""
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np

## DATA PREPARATION

mat = spio.loadmat('XwindowsDocData.mat',squeeze_me=True)
X_train = np.asarray(mat['xtrain']).tolist().todense()
y_train = np.asarray(mat['ytrain'])

(_,total_features) = X_train.shape
data_total = len(y_train)
xw_total = (y_train==1).sum() # Windows X
mw_total = (y_train==2).sum() # Microsoft Window

""" Model Fitting """

priors = [ xw_total/data_total , mw_total/data_total] # Class priors 
Njc = np.ones([2,total_features])

for i in range(0,data_total):
    c = y_train[i]-1 # python lists start from 0
    for j in range(0,total_features):
        if X_train[i,j]==1:
            Njc[c,j] += 1
Ojc = np.zeros([2,total_features])
Ojc[0] = Njc[0] / xw_total # number of occurrancies of each feature for class 1
Ojc[1] = Njc[1] / mw_total # number of occurancies of each feature for class 2

# Priors and Njc are model parameters
#plt.figure(1)
#plt.bar(Njc[0])


plt.figure(2,figsize=(15,11))
plt.bar(range(0,total_features),Ojc[0])
plt.xlabel('Features')
plt.ylabel('Probabilty')
plt.title('p(x=1|y=1) - Class 1 Conditional Probability')
plt.show()

plt.figure(3,figsize=(15,11))
plt.bar(range(0,total_features),Ojc[1])
plt.xlabel('Features')
plt.ylabel('Probabilty')
plt.title('p(x=1|y=2) - Class 2 Conditional Probability')
plt.show()




un_inf = np.isclose(Ojc[0],Ojc[1]).sum() # checks the two arrays for any element wise close
"""Number of Uninformative Features"""
print("Number of Uninformative features:",un_inf)
un_inf_m = np.where(np.isclose(Ojc[0],Ojc[1],rtol=1e-04))[0]
print(un_inf_m +1) # Uninformative features


# # Exercise 3
# Classification – discrete data

X_test = np.asarray(mat['xtest']).tolist().todense()
y_test = np.asarray(mat['ytest'])


"""MAP Estimate P(y=c|x_test,Ojc) = log(Ojc)+log(prior(c))"""

y_pred = []
for i in range(0,len(y_test)):
    # MAP ESTIMATE OF EACH TEST EXAMPLE
    map_est = np.zeros([2,1])
    for j in range(0,total_features):
        for c in range(0,2):
            map_est[c] += np.power(np.log(Ojc[c][j]),X_test[i,j])*np.power(np.log(1-Ojc[c][j]),1-X_test[i,j])
            map_est[c] += np.log(priors[c])
    predicted_label = np.argmax(map_est)+1
    y_pred.append(predicted_label)
accuracy_test = (np.sum(y_test.T==y_pred)/data_total)*100

y_pred = []
for i in range(0,len(y_train)):
    # MAP ESTIMATE OF EACH TEST EXAMPLE
    map_est = np.zeros([2,1])
    for j in range(0,total_features):
        for c in range(0,2):
            map_est[c] += np.power(np.log(Ojc[c][j]),X_train[i,j])*np.power(np.log(1-Ojc[c][j]),1-X_train[i,j])
            map_est[c] += np.log(priors[c])
    predicted_label = np.argmax(map_est)+1
    y_pred.append(predicted_label)
accuracy_train = (np.sum(y_train.T==y_pred)/data_total)*100

""" Optional Part"""
Oj = np.sum((Ojc.T*priors),axis=1)

Ij= np.zeros([1,total_features])
for c in range(0,2):
    Ij += (Ojc[c]*priors[c])*np.log(np.divide(Ojc[c],Oj+np.finfo(float).eps))
    Ij += (1-Ojc[c])*priors[c]*np.log(np.divide(1-Ojc[c],(1-Oj)+np.finfo(float).eps))
Ij=Ij[0].flatten()
Ij = np.argsort(Ij)[::-1] # index of Largest Information
k = 10
new_X_train = X_train[:,Ij[:k]]
new_y_train = y_train[Ij[:k]]



#y_pred==y_test
