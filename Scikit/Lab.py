'''
    file name: LAB
    author: Ji Woo Kim
    modified: 04.03, 2022
'''
import pandas as pd
import numpy as np
import sklearn.datasets

bmi_data = pd.read_csv('data/bmi_data_lab3.csv')

print(bmi_data.describe())
print()
print(bmi_data.dtypes)