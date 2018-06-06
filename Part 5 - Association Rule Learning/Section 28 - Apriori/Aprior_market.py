#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:48:58 2018

@author: nitishharsoor
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transaction= []

# Convert Dataset into List
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j])for j in range(0,20)])
    # Appending Rows(of 7501) for i loop and Columns(of 20) for j loop
    # []-->Brackets convert them to List
    # Values in list should be in 'string' format hence, -->str() function
    # Taking values from Dataset hence---> Dataset.values[i,j] function

# Training Apriori on the dataset
from apyori import apriori
    # apyori is open associative libarary imported as file in directory
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

