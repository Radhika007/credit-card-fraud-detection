#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:31:27 2019

@author: radhika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, 0:30 ].values
y = dataset['Class']

# Labelling the data as fraud or not_fraud
frauds = dataset.loc[dataset['Class'] == 1]
non_frauds = dataset.loc[dataset['Class'] == 0]
print("We have ", len(frauds), "frauds and" , len(non_frauds), "non-frauds")

# Splitting the dataset into test data and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Using Logistic Regression
from sklearn import linear_model
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_train,y_train)
score = logistic.score(X_test,y_test)
print("Score: ", score)

# Prediction
y_pred = np.array(logistic.predict(X_test))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



