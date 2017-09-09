# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 01:39:52 2017

@author: zaghlollight
"""
#import lib and data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#spliting dataset
from sklearn.cross_validation import train_test_split
Xtrain,Xtest,Ytrain,Ytest =train_test_split(x,y,test_size=0.25)

#feat scaling 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
Xtrain=sc_X.fit_transform(Xtrain)
Xtest=sc_X.transform(Xtest)

#train RF model
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(Xtrain,Ytrain)


#prediction new value
Ypred=classifier.predict(Xtest)

#confusion matrix for evaluate
from sklearn.metrics import confusion_matrix
cn=confusion_matrix(Ytest,Ypred)

#visualise train set
from matplotlib.colors import ListedColormap
X_set, y_set = Xtrain, Ytrain
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
l=np.unique(y_set)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('RF (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#visualise test set

X_set, y_set = Xtest, Ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('RF(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()