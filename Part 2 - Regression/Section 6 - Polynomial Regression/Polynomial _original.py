#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 19:33:20 2018

@author: nitishharsoor
"""
# *********************************************************************************************
# Truth or Bluff Project on Salary Level by future Employee(level=6.5 , Mentioned salary=160k)*
# *********************************************************************************************


# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# SPLIITING is not used BECAUSE Dataset is small of size=10 

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting ////*****Linear regression*****//// to  dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y) # just fitting variables to model NO TRANSFORM
# Linear Regressor is Just refrence for comparing results with polynomial Regression results.


# Fitting ////******Polynomial Regression*******//// model to Dataset

# "POLYNOMIAL FEATURES"-->is a Class, imported that Gives Tools to include some polynomial terms into linear regression equation.
# class Also includes One's into Column for b0 constant in the equation.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)# //***"poly_reg"***/// --> Object that Transforms independent variables(Level) into (level)x^2/3/4/5.......
X_poly = poly_reg.fit_transform(X) # fit_transform ---> Fitting model with variables(Level) of linear regession and transforming  into X_poly.
# fitting poly regression to linear regression. 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# This is done for Comparing between these model's results.


# Visualizing ////Linear Regression model///// Results
plt.scatter(X, y, color = 'blue')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualizing ////Polynomial Regression model///// Results
plt.scatter(X, y, color = 'blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'red') # Predict parameter Contain b0,b1x1^1,b2x1^2 from Level variable 
# 1. fitting and transform Because,Lin_reg_2 is in Linear not polynomial
# 2. lin_reg_2 object is used Because we are usig Linear Regressor to transform into Polynomial
# 3. refer 'poly_reg"
plt.title('Truth or Bluff(Polynomail Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualizing ////Polynomial Regression model///// Results
# (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'blue')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'red') # Predict parameter Contain b0,b1x1^1,b2x1^2 from Level variable 
# 1. fitting and transform Because,Lin_reg_2 is in Linear not polynomial
# 2. lin_reg_2 object is used Because we are usig Linear Regressor to transform into Polynomial
# 3. refer 'poly_reg"
plt.title('Truth or Bluff(Polynomail Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5) #  pedicting only one value instead of all X(level) variable values

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))