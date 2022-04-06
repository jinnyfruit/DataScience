'''
    file name: PHW 2
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


bestFeatures = SelectKBest(score_func=chi2,k=4)
fit = bestFeatures.fit(X,y)
dfColums = pd.DataFrame(feature)
dfTarget = pd.DataFrame(fit.scores_)

# Concatenate two dataframes for better visualization
featureScores = pd.concat([dfColums,dfTarget],axis=1)
featureScores.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population','AveOccup','targets']

print(featureScores.nlargest(4,'Score'))

# feature Importance Scoring
model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_,index=feature)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# Correlation Matrix with heatmap

# get corrlations of pairs
corrmat = df.corr()
top_corr_featrues = corrmat.index
plt.figure(figsize=(8,8))
g = sns.heatmap(df[top_corr_featrues].corr(),annot = True, cmap = "RdYLGn")




