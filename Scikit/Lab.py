'''
    file name: LAB
    author: Ji Woo Kim
    modified: 04.03, 2022
'''
import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn.axisgrid
from sklearn.preprocessing import MinMaxScaler

bmi_data = pd.read_csv('data/bmi_data_lab3.csv')

print(bmi_data.describe())
print()
print(bmi_data.dtypes)

bmi = bmi_data["BMI"]
plt.hist(bmi,bins=10,label='BMI rate')
plt.show()

'''
Standard_Scaler = StandardScaler()
fitted = Standard_Scaler.fit(bmi_data['Height (Inches)','Weight (Pounds)'])
print(fitted)
'''

# replacing missing or wrong data into nan
bmi_data.fillna(np.nan)
print(bmi_data)
bmi_data.replace(0,np.nan)
bmi_data.replace(bmi_data < 0, np.nan)

check_for_nan = bmi_data.isna()
print(check_for_nan)
