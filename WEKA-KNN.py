# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 03:48:53 2021

@author: samiha hussein
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



data = "C:/Users/samiha hussein/Downloads/SeoulBikeData.csv"
path1= "D:/college/Semester 6/Artificial Intelligence/ProjectAI-Weka/SeoulBikeData.arff"
# Assign colum names to the dataset
names = ['Rented Bike', 'Hour', 'Temperature', 'Humidity', 'Wind speed',
         'Visibility', 'Dew point temperature', 'Solar Radiation',
         'Rainfall','Snowfall','Seasons','Holiday','Functioning Day']

# Read dataset to pandas dataframe
#encoding='latin1' used while using arff
dataset = pd.read_csv(data, encoding="latin1" ,names=names)

#dataset.head()
print(dataset.head())

#split our dataset into its attributes and labels
X = dataset.iloc[:, :-1].values  #columns
y = dataset.iloc[:, 4].values   #labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


#The following script performs feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)














