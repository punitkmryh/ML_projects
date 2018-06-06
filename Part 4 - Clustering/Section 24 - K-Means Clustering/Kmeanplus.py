#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:02:40 2018

@author: nitishharsoor
"""
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Datasets using pandas
dataset= pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values#Considered as X-axis

#using Elbow method to find Optimal NO of Clusters

from sklearn.cluster import KMeans
wcss=[]# Considered as Y axis 
for i in range(1,11):
    kmeans= KMeans(n_clusters=i, init='k-means++', max_iter=300 ,n_init = 10 ,random_state=0)
    kmeans.fit(X)   
    wcss.append(kmeans.inertia_)#interia package is Not but Sum of Sq(Distance from 2 points in clusters)

plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('No of clusters ')
plt.ylabel('Wcss Result values')
plt.show()

# Appling Kmeans to Mall_Customers datasets to predict which cluster they belong 
# Cluster will range from 0 to 4(total=5) 
kmeans = KMeans(n_clusters =5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualizing Clusters of 5 sets

    #////****Creating Clusters on graph*****/////
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=50,c='red',label='C1:High Income ,low spending->Carefull')
    # plt.Scatter(x-axis,y-axis) 
    # here X-axis--> X[y-kmeans=cluster No, Customers income]
    # and y_axis--> X[y_kmeans = cluster no, Customers Spending]
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=50,c='blue',label='c2:Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=50,c='Yellow',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=50,c='green',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=50,c='cyan',label='Sensible')
    # Ploting CENTROIDS for Clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='Black',label='Centroids')
# x-axis->>for Column 1 of X
# y-axis->>for column  2 of X
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()# specified all labels so lengend
plt.show()

