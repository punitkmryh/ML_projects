#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:22:09 2018

@author: nitishharsoor
"""
# importing libararies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing Datasets using pandas
dataset= pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values#Considered as X-axis

# using Dendrogram alo to find No of optimal Clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('No. of customers')
plt.ylabel('Eulidean distance')
plt.show()

# fitting hierarchical Clustering to mall_customers dataset
from sklearn.cluster import AgglomerativeClustering
hc =  AgglomerativeClustering(n_clusters = 5,affinity='euclidean',linkage='ward')
# 'ward' linkage_method--> minimize the variance in each clusters
y_hc = hc.fit_predict(X)

#Visualizing Hierarchical Clusters of 5 sets

    #////****Creating Clusters on graph*****/////
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=50,c='red',label='C1:High Income ,low spending->Carefull')
    # plt.Scatter(x-axis,y-axis) 
    # here X-axis--> X[y-kmeans=cluster No, Customers income]
    # and y_axis--> X[y_hc = cluster no, Customers Spending]
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=50,c='blue',label='c2:Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=50,c='Yellow',label='Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=50,c='green',label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=50,c='cyan',label='Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()# specified all labels so lengend
plt.show()