
import numpy as np
import pandas as pd  # data processing, CSV file I/O

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('SeoulBikeData.csv')
col_names = ['Date','Rented_Bike_Count', 'Hour', 'Temperature(C)', 'Humidity(%)',
             'Wind_Speed(m/s)', 'Visibility(10m)', 'Dew_Point_Temperature(C)', 'Solar_Radiation(MJ/m2)', 'Rainfall',
             'Snowfall', 'Seasons', 'Holiday', 'Functioning_Day']
dataset.columns = col_names
print(len(dataset))
# print(dataset.head())

# print(dataset.info())


#Splitting the dataset into attributes and labels
X = dataset.iloc[:, [1, 2, 3]].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
# print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
