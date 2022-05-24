'''
file name: Lab3
author: Ji Woo Kim
modified: 2022.05.18
'''
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import KFold
from numpy import array
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Read dataset
df = pd.read_csv("mnist.csv")
print("\n------------------------------mnist.csv------------------------------\n")
print(df.head())

# Check if the data has any NAN data
print("\n----------------------Find nan data----------------------\n")
print(df.isnull().any())
print("\n------------------------------Drop nan data------------------------------\n")
df.dropna(inplace=True)
print(df)

# Split dataset into train/test data
X = df.drop(columns='label')
y = df['label'].values
print('\n------------------------------x - print------------------------------\n')
print(X)
print("\n----------------------y - print----------------------\n")
print(y)

# Prepare cross validation
kfold = KFold(5,shuffle=True,random_state=46)

for train_x,test_y in kfold.split(X):
  X_train = pd.DataFrame(X.iloc[train_x,:])
  X_test = pd.DataFrame(X.iloc[test_y,:])

print("\n----------------------X train split result----------------------\n")
print(X_train)
print("\n----------------------X test split result----------------------\n")
print(X_test)

for train_y,test_y in kfold.split(y):
  Y_train = y[train_y]
  Y_test = y[test_y]

print("\n----------------------Y train split result----------------------\n")
print(Y_train)
print("\n----------------------Y test split result----------------------\n")
print(Y_test)

# Create a KNN classifier
knn_model1 = KNeighborsClassifier(n_neighbors=3)
knn_model2 = KNeighborsClassifier(n_neighbors=5)
knn_model3 = KNeighborsClassifier()

# Train the KNN classifier
knn_model1.fit(X_train,Y_train)
knn_model2.fit(X_train,Y_train)

# Predict the result
knn_model1.predict(X_test)
knn_model2.predict(X_test)

# Get the score
print("\n---------------3 neighbors score---------------\n")
print(knn_model1.score(X_test,Y_test))
print("\n---------------5 neighbors score---------------\n")
print(knn_model2.score(X_test,Y_test))

# Create a dictionary of all values
parameter_grid = {'n_neighbors': np.arange(1,25)}

# Test all values for n_neighbors and fit the data
knn_gscv = GridSearchCV(knn_model3,parameter_grid,cv = 5)
knn_gscv.fit(X_train,Y_train)

best_parameter = knn_gscv.best_params_
print("\n---------------best parameter---------------\n")
print(best_parameter)

print("\n---------------gscv best score---------------\n")
print(knn_gscv.best_score_)










