'''
    file name: PHW 3
    author: Ji Woo Kim
    modified: 04.02, 2022
'''
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
import matplotlib.pylab as plt
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.preprocessing import  StandardScaler
from sklearn.decomposition import PCA

dataset = sklearn.datasets.fetch_california_housing(data_home = None, download_if_missing = True, return_X_y = False, as_frame = False)
# print(dataset.DESCR)

df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
feature = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population','AveOccup']
df['target'] = dataset.target

# Separation of data
X = df.loc[:,feature].values
print(X)
y = df.loc[:,['target']].values
y = y.astype('int')
print(y)


X = StandardScaler().fit_transform(X)
pca  = PCA(n_components=2)
printcipalComponents = pca.fit_transform(X)
principalDF = pd.DataFrame(printcipalComponents,['principal componet1','principal component2'])
finalDF = pd.concat([principalDF,df[['target']]],axis=1)
pca.explained_variance_ratio_




