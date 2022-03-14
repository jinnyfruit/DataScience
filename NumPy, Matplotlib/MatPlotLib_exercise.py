# file name: MatPlotLib Coding Exercise
# author: 202035513 Ji Woo Kim
# modified: 2022.03.14
import numpy as np
import matplotlib.pyplot as plt

wt = np.random.uniform(low=40.0, high=90.0, size=100)
ht = np.random.randint(low=140, high=200, size=100)

num_s = len(wt)
BMI = np.zeros(100)
UnderWeight = 0
Healthy = 0
Overweight = 0
Obese = 0

# Calculate BMI and store it in a new array
for i in range(num_s):
    BMI[i] = wt[i] / ((ht[i] * ht[i])/10000)
    if BMI[i] < 18.5:
        UnderWeight += 1
    elif (18.5 <= BMI[i]) & (BMI[i] < 24.9):
        Healthy += 1
    elif (25 <= BMI[i]) & (BMI[i] < 29.9):
        Overweight += 1
    else:
        Obese += 1

# Bar Chart
plt.figure(1)
BMI_classification = ['UnderWeight', 'Healthy', 'Overweight', 'Obese']
Num_Of_S = [UnderWeight, Healthy, Overweight, Obese]
plt.bar(BMI_classification, Num_Of_S)

# Histogram
plt.figure(2)
plt.hist(BMI, bins=[0, 18.5, 25.0, 30.0, 45.0])
plt.title("Histogram of students BMI")
plt.xticks([0, 18.5, 25.0, 30.0, 45.0])
plt.xlabel('value of BMI')
plt.ylabel('number of students')

# Pie Chart
plt.figure(3)
plt.pie(Num_Of_S, labels=BMI_classification, autopct='%1.2f%%')

# Scatter Plot
plt.figure(4)
Scatter_Range = np.arange(100)
plt.scatter(Scatter_Range, wt)
plt.scatter(Scatter_Range, ht)
plt.xlabel('Weight')
plt.ylabel('height')
plt.title('Scatter plot')

plt.show()
