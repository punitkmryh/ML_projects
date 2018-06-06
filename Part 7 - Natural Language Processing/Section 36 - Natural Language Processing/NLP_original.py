#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:23:57 2018

@author: nitishharsoor
"""
# Step:1 Importing Libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# step: 2 Importing the ///Tab seperated Values(TSV)/// using Panadas.
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# step: 3 Cleaning/steming the Text 
import re
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords# Cleaning library
from nltk.stem.porter import PorterStemmer# Stemming Library

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    # *step: 4 **//cleaning//*** 
    #   text becz,During "Bag of Words Model"/representation,
    #   it will only use the "Relevent Words" like --> woow,delicious etc 
    #   rather than words like ---> the,um,an,capitals,Puncuations, etc 
    review = review.split()# Split sententse into words,which is list of Words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]# transforming "list" into "SET" which is faster way to search
    # step: 5 ***//Stemming//***
    #   Used to Avoid To many Words like "loved","loving" 
    #   into a regroup of Version of Single word "Love" which is root word used in Future.
    #   it reduces the sparax matrices space
    
    # step: 7 Reverse String (from Set(review) to string(review)) by --->joining funcation
    review = ' '.join(review)
    corpus.append(review)
    
# Step 8: Creating Bag of Words model
    # Just to obtain a classification model in future
    # used for Simplify all the review with minimum No.of words
    # tokenization-->Sparse matrix(bag of words model)-->less zeros
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)

# Step 9: Creating Independent Variable
X = cv.fit_transform(corpus).toarray() 

# Step 10: Creating dependent Variable
y = dataset.iloc[:,1].values

# Step 11: ***//Classification//***
    # Based on experiance NLP best fits for //Naive Bayes, Decision Tree, Random Forest//
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Choose more Datasets for training and less to test,bcz of 1500 datasets

# NO Need of Feature Scaling becz, most of them are Zero and 1
 
    #Lets use Naive bayes
from sklearn.naive_bayes import GaussianNB
classifier =  GaussianNB()
classifier.fit(X_train,y_train)

#predicting Test Results 
y_pred = classifier.predict(X_test)

# confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#calculating Accuracy
(55+91)/200=.73 (accuracy)






