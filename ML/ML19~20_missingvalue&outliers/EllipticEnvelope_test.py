import pandas as pd,numpy as np
from sklearn.covariance import EllipticEnvelope

path = "D:\Project\Kaggle_Project\\bike/"   

train = pd.read_csv(path + 'train.csv')

x = train.drop(['datetime'],axis=1)
x = x[['temp','humidity']]
x = x[:20]
print(x[:1])

col_list = x.columns

outliers = EllipticEnvelope(contamination=.2)

for i in col_list:
    a = np.array(x[f'{i}']).reshape(-1,1)
    outliers.fit(a)
    pred = outliers.predict(a)
    print(i)
    print(pred)
    b = list(pred)
    print(b.count(-1))
    index_for_outlier = np.where(pred == -1)
    print('outier indexex are', index_for_outlier)
    outlier_value = a[index_for_outlier]
    print('outlier_value :', outlier_value)
    