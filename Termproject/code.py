'''
file name : Term Project - Regression of of online learning adaptability according to online experience
modified: 2022.05.01
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Regression of of online learning adaptability according to online experience

# read the data
df = pd.read_csv('covid_student.csv')
print(df)

# check dataframe
# print(df.describe())

# change categorical data into numerical data

# drop dirty data

# get columns from the df
Online_Class_Spent_Time = df['Time spent on Online Class']
Online_Class_Experience = df['Rating of Online Class experience']
Self_study = df['Time spent on self study']
SNS_spent_time = df['Time spent on social media']
TV_spent_time = df['Time spent on TV']
# get online total time

# data split
print(df.describe())

# make a model

# train a model

# test a model

# model accuracy




