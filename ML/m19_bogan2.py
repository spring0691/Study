import numpy as np, pandas as pd

date = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4,np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

print(date)