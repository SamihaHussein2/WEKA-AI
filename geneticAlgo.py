# tpot is for machine learning models
from tpot import TPOT

from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

#Load data
dataset=pd.read_csv('SeoulBikeData.csv')

#Clean data
dataset_shuffle=dataset.iloc[np.random.permutation(len(dataset))]
data=dataset_shuffle.reset_index(drop=True)

#Store 2 classes
data['Class']=data['Class'].map({'Yes':1, 'No':0})
data_class=data['Class'].values

#Split training,testing, and validation data
training_indices, validation_indices=training_indices,testing_indices=train_test_split(data.index, stratify=data_class, train_size=0.75,test_size=0.25)

#Genetic programming finding the best ML model and hyperparameters
tpot=TPOT(generations=5, vebrosity=2)
tpot.fit(data.drop('Class',axis=1).loc[training_indices].values,
data.loc[training_indices, 'Class'].values)

#Score the accuracy
tpot.score(data.drop('Class',axis=1).loc[validation_indices].values,
           data.loc[validation_indices,'Class'].values)

#Export the code
tpot.export('pipeline.py')

