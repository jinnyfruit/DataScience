'''
file name: decision Tree
author: Ji Woo Kim
modified: 2022.05.09
'''
import numpy as np
import pandas as pd
from math import sqrt

# read data
dataset = pd.read_csv('TshirtData.csv')
dataset.replace('M',0, inplace=True)
dataset.replace('L',1, inplace=True)
x = dataset['irl.']
y = dataset['Weight']
print(dataset)

# calculate the Euclidean distance between two vectors
# row = [x, y, type]
def euclidean_distance(x1, y1,x2,y2):
	distance = 0.0
	distance = sqrt((x1-x2)**2 + (y1-y2)**2)

	return distance

# Locate the most similar neighbors
def get_neighbors(x,y, x1,y1, num_neighbors):
	distances = list()
	for i in range(len(x)):
		dist = euclidean_distance(x[i],y[i],x1,y1)
		distances.append((dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

neighbors = get_neighbors(x,y, 161,61,len(x))
for neighbor in neighbors:
	print(neighbor)