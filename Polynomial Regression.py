# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:36:54 2019

@author: Sammietex
"""

#Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#splitting the datasets into the Training set and Test set\
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
