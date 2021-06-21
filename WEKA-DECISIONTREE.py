# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 06:39:20 2021

@author: samiha hussein
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#path of the dataset
filepath = 'SeoulBikeData.csv'

#styling a header with labels
names = ['Rented Bike', 'Hour', 'Temperature', 'Humidity', 'Wind speed',
         'Visibility', 'Dew point temperature', 'Solar Radiation',
         'Rainfall','Snowfall','Seasons','Holiday','Functioning Day']

#reading csv file containing data
data = pd.read_csv(filepath ,encoding="latin1", header=0)

#displaying data lenght
print("Data Lenght:" , len(data))

# displaying how many columns and rows
print("Dataset Shape:", data.shape  )

#printing the dataset
print("dataset:") 
print(data.head())

#Seperating target variables
X = data.values[:,1:5]
Y = data.values[:,0]
#splitting dataset into test and train
X_train, X_test , y_train , y_test = train_test_split(X,Y,test_size=0.3,random_state=100)

#function to perform training with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy",random_state=100,
max_depth=3 , min_samples_leaf=3)
clf_entropy.fit(X_train,y_train)

#function to make predections
y_pred_en = clf_entropy.predict(X_test)
print("Predictions: ")
print(y_pred_en)

#checking accuracy 
print("Accuracy is: ", accuracy_score(y_test, y_pred_en)*100)



