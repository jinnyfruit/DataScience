#KNN
import pandas as pd
import numpy as np
Tshirt = pd.read_csv("/Users/jeongdeok/Downloads/Tshirt.csv")

x = Tshirt.iloc[ :, 0 : 2 ].copy()
#copy without Tsize

mean_H = Tshirt['Height'].mean()
mean_W = Tshirt['Weight'].mean()
std_H = np.std(x,axis = 0)['Height']
std_W = np.std(x,axis = 0)['Weight']
#mean, std score

std_data = (x - np.mean(x,axis = 0))/ np.std(x,axis = 0)
#Pre-processing using standard score

input = pd.DataFrame([[161, 61, None]])
#Data to predict

dist =[]
for i in Tshirt.index:
    line = []
    line.append(i)
    line.append( np.sqrt((input.loc[0][0] - Tshirt.loc[i]['Height'])**2 + (input.loc[0][1] - Tshirt.loc[i]['Weight'])**2))
    dist.append(line)
#Store the distance of points in the list

dist.sort(key=lambda x:x[1])
#Sort in close order

k = 3
M = 0
L = 0
for i in range(k):
    if Tshirt.loc[dist[i][0]]['Tsize'] == 'L':
        L += 1
    else:
        M += 1
if L > M:
    print('predict : L')
else:
    print('predict : M')
#Read as many as k and predict
