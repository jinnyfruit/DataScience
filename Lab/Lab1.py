'''
file name: Lab1
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
from sklearn.metrics import mean_squared_error

# Read dataset
df = pd.read_csv("housing.csv")
print("\n------------------------------Head housing.csv------------------------------\n")
print(df)

# Check if the data has any NAN data
print("\n----------------------Find nan data----------------------\n")
print(df.isnull().any())
print("\n------------------------------Drop nan data------------------------------\n")
df.dropna(inplace=True)
print(df)

# Split dataset into train/test data
x = df.drop(columns=['median_house_value','ocean_proximity'])
y = df['median_house_value'].values
print('\n------------------------------x - print------------------------------\n')
print(x)
print("\n----------------------y - print----------------------\n")
print(y)

# first split test data 1:5
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, stratify = None, shuffle=True, test_size=0.2, random_state=42)
print("\n------------------------------x_train1------------------------------\n")
print(x_train1)
print("\n------------------------------x_test1------------------------------\n")
print(x_test1)
print("\n----------------------y_train1----------------------\n")
print(y_train1)
print("\n----------------------y_test1----------------------")
print(y_test1)

# second split test data 2:5
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, stratify = None, shuffle=True, test_size=0.4, random_state=42)
print("\n------------------------------x_train2------------------------------\n")
print(x_train2)
print("\n------------------------------x_test2------------------------------\n")
print(x_test2)
print("\n----------------------y_train2----------------------\n")
print(y_train2)
print("\n----------------------y_test2----------------------")
print(y_test2)

# Make linear Regression model and fit data
regression_model = LinearRegression()
regression_model.fit(x_train1, y_train1)
regression_model.fit(x_train2, y_train2)

# Get the prediction
y_predict1 = regression_model.predict(x_test1)
y_predict2 = regression_model.predict(x_test2)

# Calculate the accuracy
print("\n----------------1:5 Accuracy----------------")
print(round(regression_model.score(x_test1, y_test1), 3))
print("\n----------------2:5 Accuracy----------------")
print(round(regression_model.score(x_test2, y_test2), 3))



