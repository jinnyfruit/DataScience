import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print("***** Train_Set *****")
print(train.describe())
print("\n")
print(test.describe())
print("\n")

# Get Statistics on the Missing values

# For the train set
train.isna().head()

# For the test set
test.isna().head()

print("***** In the train set *****")
print(train.isna().sum())
print("\n")

print("***** In the test set *****")
print(test.isna().sum())
print("\n")

# Fill missing values

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)

# Fill missing values with mean column values in the test set
train.fillna(test.mean(), inplace=True)

print(train.isna().sum())
print("\n")
print(train.isna().sum())
print("\n")