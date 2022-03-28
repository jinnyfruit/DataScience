"""
name: Ji Woo Kim
modified: 2022.03.22
"""
import pandas as pd
import numpy as np

# Creating an Array in NumPy
arr = np.array([[3., '?', 2., 5.],
               ['*', 4., 5., 6.],
               ['+', 3., 2., '&'],
               [5., '?', 7., '!']])

# Convert numpy array to pandas dataframe
df_ar = pd.DataFrame(arr)
print(df_ar)
print()

# Convert non-numeric to NaN
df_ar = df_ar.replace(['?', '*', '+', '!', '&'], np.nan)
print(df_ar)
print()

# any and sum of the isna method
df_ar_test = df_ar.isna().any()
print(df_ar_test)
print()

df_ar_test = df_ar.isna().sum()
print(df_ar_test)
print()

# Various properties of dropna method
df_ar_test = df_ar.dropna(axis=1, how='any')
print(df_ar_test)
print()

df_ar_test = df_ar.dropna(axis=1, how='all')
print(df_ar_test)
print()

df_ar_test = df_ar.dropna(axis=1, thresh=1)
print(df_ar_test)
print()

df_ar_test = df_ar.dropna(axis=1, thresh=2)
print(df_ar_test)
print()

# Various properties of fillna method
df_ar_test = df_ar.fillna(100.)
df_ar_test = df_ar_test.astype(float)   # Failure to do so may result in many errors.
print(df_ar_test)
print()

mean = df_ar_test.mean()
print(mean)
print()

median = df_ar_test.median()
print(median)
print()

# ffill and bfill output
df_ar_test = df_ar.fillna(method='ffill')
print(df_ar_test)
print()

df_ar_test = df_ar.fillna(method='bfill')
print(df_ar_test)
print()
