import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

def least_square_regression_line(x, y):
    b = (np.mean(x) * np.mean(y) - np.mean(x * y)) / (np.mean(x) * np.mean(x) - np.mean(x * x))
    a = np.mean(y) - np.mean(x) * b
    return a, b

# setting data
spends = np.array([2400, 2650, 2350, 4950, 3100, 2500, 5106, 3100, 2900, 1750])
income = np.array([41200, 50100, 52000, 66000, 44500, 37700, 73500, 37500, 56700, 35600])

# get a linear line
a, b = least_square_regression_line(spends, income)
print(f'y = {a:.2f}+{b:.2f}x')

# drow a dataset and line
plt.plot(spends, a + b * spends)
plt.scatter(spends, income)
plt.show()