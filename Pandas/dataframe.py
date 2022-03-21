import pandas as pd
import numpy as np
df = pd.DataFrame({'column_a': [1, 2, 4, 4, np.nan, np.nan, 6],
                   'column_b': [1.2, 1.4, np.nan, 6.2, None, 1.1, 4.3],
                   'column_c': ['a', '?', 'c', 'd', '--', np.nan, 'd'],
                   'column_d': [True, True, np.nan, None, False, True, False]})

print(df)