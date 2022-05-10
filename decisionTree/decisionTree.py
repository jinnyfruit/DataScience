'''
file name: decision Tree
author: Ji Woo Kim
modified: 2022.05.09
'''
import numpy as np
import pandas as pd
from pprint import pprint
import math

# read data
df = pd.read_csv('Marketing_response.csv')
features = df[['District','House Type','Income','Previous Customer','Outcome']]
target = df['Outcome']

# get entropy
def entropy(target_idx):
    elements, counts = np.unique(target_idx, return_counts = True)
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# get information gain
def InfoGain(data, att_name, target_name):

    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[att_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) *
                               entropy(data.where(data[att_name] == vals[i]).dropna()[target_name])
                               for i in range(len(vals))])

    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

# make a tree
def MakeTree(data, original_value, features, target_attribute_name, parent_node_class = None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(original_value[target_attribute_name]) \
            [np.argmax(np.unique(original_value[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name]) \
            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # make tree
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = MakeTree(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return (tree)

# get the result of data,information gain, tree result
print("---------------------------- data ----------------------------")
print(df)
print("\n")

print("----------------------- Information Gain -----------------------")
print('info_Gain(District) = ', round(InfoGain(df,"District","Outcome"),2),'\n')
print('info_Gain(House Type) = ', round(InfoGain(df,"House Type","Outcome"),2),'\n')
print('info_Gain(Income) = ', round(InfoGain(df,"Income","Outcome"),2),'\n')
print('info_Gain(Previous Customer) = ', round(InfoGain(df,"Previous Customer","Outcome"),2),'\n')

print("----------------------- Tree result -----------------------")
tree = MakeTree(df, df, ["District", "House Type", "Income", "Previous Customer"], "Outcome")
pprint(tree)


