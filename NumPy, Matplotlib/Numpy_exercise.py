# file name: NumPy Coding Exercise
# author: 202035513 Ji Woo Kim
# modified: 2022.03.14

import numpy as np
# Generate the height and weight of 100 students using random numbers
wt = np.random.uniform(low=40.0, high=90.0, size=100)
ht = np.random.randint(low=140, high=200, size=100)

num_s = len(wt)
BMI = np.zeros(100)

print("The number of students to calculate BMI is", num_s)
print()

# Calculate BMI and store it in a new array
for i in range(num_s):
    BMI[i] = wt[i] / ((ht[i] * ht[i])/10000)


