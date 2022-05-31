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
df = pd.read_csv("covid_students_survay.csv")
print("\n------------------------------Head covid_students_survay.csv------------------------------\n")
print(df)

# Check if the data has any NAN data
print("\n----------------------Find nan data----------------------\n")
print(df.isnull().any())
print("\n------------------------------Drop nan data------------------------------\n")
df.dropna(inplace=True)
print(df)
df = df.replace({'On (All the time)':100,'Always On':100,'On (Always On)': 100, 'On (To answer Teacher)': 50,'On (When teacher asks)':50,'Off (Untidy appearance)':0,'Off (Shy to switch on)':0,'Off (No mic available)':50,'Off (No webcam available)':50,'Off (Other reason)':-50,'Off (Other reason)': -50, 'Off (Do not want to reply)':-50})
df = df.replace({'No': -50,'Yes':50})
df.dropna(inplace=True)

# Split dataset into train/test data 'Webcam status during class','Are you interested in attending online classes?' 'Mic status during class'
x = df[['Webcam status during class','Mic status during class','Are you interested in attending online classes?']]
y = df['Are you able to understand the concepts through online classes?'].values
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

# Make linear Regression model and fit data
regression_model = LinearRegression()
regression_model.fit(x_train1, y_train1)

# Get the prediction
y_predict1 = regression_model.predict(x_test1)

# Calculate the accuracy
print("\n----------------Model Accuracy----------------")
print(round(regression_model.score(x_test1, y_test1), 3))




