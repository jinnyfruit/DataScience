'''
file name: Grid Search
name: Kim Ji Woo
modified: 2022.05.26
'''
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as np
import numpy as np

# Read data
df = pd.read_csv("wineQualityReds.csv",sep=',')

# Data Analysis
print("\n---------------------------data curation---------------------------")
print(df.head())

# Data Preprocessing
X = df.iloc[:, 0:11].values
y = df.iloc[:, 11].values

print("\n---------------------------X---------------------------")
print(X)
print("\n---------------------------Y---------------------------")
print(y)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Data Scaling
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

# Define Classifier and train
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
all_accuracies = cross_val_score(classifier, X = X_train, y = y_train, cv = 5)

print("\n---------------------------data accuracies---------------------------")
print(all_accuracies)

# Define grid parameters
grid_param = {'n_estimators':[100,300,500,800,1000],'criterion':['gini','entropy'],'bootstrap':[True, False]}

# Initialize the GridSearch CV Class
#grid_search = GridSearchCV(estimator = classifier, param_grid= grid_param, scoring='accuracy',cv = 5, n_jobs=-1)

# Train the class
#grid_search.fit(X_train, y_train
