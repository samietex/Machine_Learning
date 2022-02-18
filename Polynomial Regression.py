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
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#splitting the datasets into the Training set and Test set\
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting linear regression to the datasets
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial regresssion to the datasets
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising the linear regresson results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()

#Visualising the polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff(Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()

#Predicting a new result with linear regression
lin_reg.predict(6.5)

#Predicting a new result with Polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))