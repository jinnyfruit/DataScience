import numpy as np
import pandas as pd

df = pd.DataFrame({'column_a': [1, 2, 4, 4, np.nan, np.nan, 6],
                   'column_b': [1.2, 1.4, np.nan, 6.2, None, 1.1, 4.3],
                   'column_c': ['a', '?', 'c', 'd', '--', np.nan, 'd'],
                   'column_d': [True, True, np.nan, None, False, True, False]})

new = pd.Series([1, 2, np.nan, 5], dtype=pd.Int64Dtype())
df['column_e'] = new

df.isna()  # show if it is missing or not

print(df.isna())
print()

df.isna().any()     # returns a boolean value for each column

print(df.isna().any())
print()

df.isna().sum()     # returns a number of missing values in each column

print(df.isna().sum())
print()
