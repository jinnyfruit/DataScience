'''
file name: Coding HW4 - Standard Scaling
author: Ji Woo Kim
modified: 2022.03.29
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import seaborn.axisgrid
from sklearn.preprocessing import StandardScaler

Standard_Scaler = StandardScaler()

# set the data.
df = np.array([28, 35, 26, 32, 28, 28, 35, 34, 46, 42, 37])

print("-----Data standardization-----")
print("original data:", df)
print("mean:", df.mean())
print("Standard Deviation:", df.std())

# Standard scaling of the data.
scaled_df = Standard_Scaler.fit_transform(df.reshape(-1,1))

print("Standard Score:", scaled_df.round(2))

idx = np.array(np.where(scaled_df < -1))

print("Student who got ")
for i in idx[0]:
    print(df[i])
print("is F.")
