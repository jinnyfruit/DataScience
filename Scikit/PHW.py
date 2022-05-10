import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model

# Data Ecploration

# read excel file and get data
df = pd.read_excel('data/bmi_data_phw3.xlsx')

# print dataset stastical data, feature names & data types
print(df.describe())
print(df[df.columns].dtypes)

# plot height&weight histograms for each BMI value
for i in range(0,5):
  hist_w1=df[df['BMI']==i].hist(column='Height (Inches)',bins=10)
  hist_h1=df[df['BMI']==i].hist(column='Weight (Pounds)',bins=10)

#scale height and weight

#using StandardScaler
  #scaler=preprocessing.StandardScaler()
  #k='Standard'
#using MinMaxScaler
  #scaler=preprocessing.MinMaxScaler()
  #k='MinMax'
#using RobustScaler
scaler=preprocessing.RobustScaler()
k='Robust'
scaled_w=scaler.fit_transform(df['Weight (Pounds)'].to_frame())
scaled_h=scaler.fit_transform(df['Height (Inches)'].to_frame())

#plot histogram of scaled data
plt.hist(scaled_h,bins=10)
plt.title(k+" Scaled Height (Inches) Histogram")
plt.show()
plt.hist(scaled_w,bins=10)
plt.title(k+" Scaled Weight (Pounds) HIstogram")
plt.show()

#scale height and weight

#using StandardScaler
  #scaler=preprocessing.StandardScaler()
  #k='Standard'
#using MinMaxScaler
  #scaler=preprocessing.MinMaxScaler()
  #k='MinMax'
#using RobustScaler
scaler=preprocessing.RobustScaler()
k='Robust'
scaled_w=scaler.fit_transform(df['Weight (Pounds)'].to_frame())
scaled_h=scaler.fit_transform(df['Height (Inches)'].to_frame())

#plot histogram of scaled data
plt.hist(scaled_h,bins=10)
plt.title(k+" Scaled Height (Inches) Histogram")
plt.show()
plt.hist(scaled_w,bins=10)
plt.title(k+" Scaled Weight (Pounds) HIstogram")
plt.show()

#divide dataset into two groups according to gender
D_m=df[df['Sex']=='Male'] #male group
D_f=df[df['Sex']=='Female'] #female group

#predict BMI with values of male group
reg_m= linear_model.LinearRegression()
hw_m=np.array(D_m.loc[:,['Height (Inches)','Weight (Pounds)']])
BMI_m=np.array(D_m['BMI'])
reg_m.fit(hw_m,BMI_m) #fit linear model
pBMI_m=reg_m.predict(hw_m) #predicted BMI value through weight and height of female group

#predict BMI with values of female group
reg_f= linear_model.LinearRegression()
hw_f=np.array(D_f.loc[:,['Height (Inches)','Weight (Pounds)']])
BMI_f=np.array(D_f['BMI'])
reg_f.fit(hw_f,BMI_f) #fit linear model
pBMI_f=reg_f.predict(hw_f) #predicted BMI value through weight and height of female group

#compare BMI estimates and actual BMI by scatter plot

#in male group
plt.title('Estimate BMI of Male group')
plt.scatter(np.arange(0,np.size(BMI_m)),BMI_m,color='r',s=10,label='actual BMI')
plt.scatter(np.arange(0,np.size(pBMI_m)),pBMI_m,color='b',s=10,label='estimated BMI')
plt.legend()
plt.show()  #x: index of data in D_m, y=BMI

#in female group
plt.title('Estimate BMI of Female group')
plt.scatter(np.arange(0,np.size(BMI_f)),BMI_f,color='r',s=10,label='actual BMI')
plt.scatter(np.arange(0,np.size(pBMI_f)),pBMI_f,color='b',s=10,label='estimated BMI')
plt.legend()
plt.show()  #x: index of data in D_m, y=BMI



