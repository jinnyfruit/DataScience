import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print("***** Train_Set *****")
print(train.head())     # print top n rows, n = 5 default
print("\n")
print("***** Test_Set *****")
print(test.head())