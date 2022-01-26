import numpy as np
import pandas as pd

aaa = np.array([[1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600,7, 800, 900, 190, 1001, 1002, 99]])
aaa = np.transpose(aaa)

# df = pd.DataFrame(aaa, columns=['x','y'])

# data1 = df[['x']]
# data2 = df[['y']]

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.15)
pred = outliers.fit_predict(aaa)
print(pred.shape) # (13,)

b = list(pred)
print(b.count(-1))
index_for_outlier = np.where(pred == -1)
print('outier indexex are', index_for_outlier)
outlier_value = aaa[index_for_outlier]
print('outlier_value :', outlier_value)