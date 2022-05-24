'''
file name: Lab2
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
from sklearn import tree

# Read dataset
df = pd.read_csv("winequality-red.csv", sep=";")
print("\n------------------------------winequality-red.csv------------------------------\n")
print(df.head())

# Check if the data has any NAN data
print("\n----------------------Find nan data----------------------\n")
print(df.isnull().any())
print("\n------------------------------Drop nan data------------------------------\n")
df.dropna(inplace=True)
print(df)

# Split dataset into train/test data
x = df.drop(columns=['quality'])
y = df['quality'].values
print('\n------------------------------x - print------------------------------\n')
print(x)
print("\n----------------------y - print----------------------\n")
print(y)

# first split test data 1:10
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, stratify = None, shuffle=True, test_size=0.1, random_state=42)
print("\n------------------------------x_train1------------------------------\n")
print(x_train1)
print("\n------------------------------x_test1------------------------------\n")
print(x_test1)
print("\n----------------------y_train1----------------------\n")
print(y_train1)
print("\n----------------------y_test1----------------------")
print(y_test1)

# second split test data 2:10
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, stratify = None, shuffle=True, test_size=0.2, random_state=42)
print("\n------------------------------x_train2------------------------------\n")
print(x_train2)
print("\n------------------------------x_test2------------------------------\n")
print(x_test2)
print("\n----------------------y_train2----------------------\n")
print(y_train2)
print("\n----------------------y_test2----------------------")
print(y_test2)

# third split test data 3:10
x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, stratify = None, shuffle=True, test_size=0.3, random_state=42)
print("\n------------------------------x_train3------------------------------\n")
print(x_train3)
print("\n------------------------------x_test3------------------------------\n")
print(x_test3)
print("\n----------------------y_train3----------------------\n")
print(y_train3)
print("\n----------------------y_test3----------------------")
print(y_test3)

# Make Tree classification model and fit data
tree_model = tree.DecisionTreeClassifier()

tree_model1 = tree_model.fit(x_train1,y_train1)
tree_model2 = tree_model.fit(x_train2,y_train2)
tree_model3 = tree_model.fit(x_train3,y_train3)

# Get the prediction
y_predict1 = tree_model.predict(x_test1)
y_predict2 = tree_model.predict(x_test2)
y_predict3 = tree_model.predict(x_test3)

# Calculate the accuracy
print("\n----------------1:9 Accuracy----------------")
print(round(tree_model.score(x_test1, y_test1), 3))
print("\n----------------2:8 Accuracy----------------")
print(round(tree_model.score(x_test2, y_test2), 3))
print("\n----------------3:7 Accuracy----------------")
print(round(tree_model.score(x_test3, y_test3), 3))

# Showing Tree
tree.plot_tree(tree_model1)
tree.plot_tree(tree_model2)
tree.plot_tree(tree_model3)
plt.show()


