import numpy as np, pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4,np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

#print(data.shape)   # (4,5)
data = data.transpose()
data.columns = ['a','b','c','d']
# print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer, IterativeImputer    # 결측치처리 도와주는넘

# imputer = SimpleImputer(strategy='constant',fill_value=777)
imputer = SimpleImputer()         
# stratery = 전략 mean 평균,median 중위값most_frequent 가장많이 사용한, 'constant',fill_value=777 원하는 상수 사용.
# fill_value만 단독사용 가능 + 먼저 우선순위에 있다.

#print(type(data))       # <class 'pandas.core.frame.DataFrame'>
# imputer.fit(data)
# data2 = imputer.transform(data)
# print(data2)

'''     <class 'numpy.ndarray'> 로 돌려준다.
[[ 2.          2.          7.33333333  6.        ]
 [ 6.66666667  4.          4.          4.        ]
 [ 6.66666667  4.66666667  7.33333333  6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
 '''
 
#print(type(data['a']))  # <class 'pandas.core.series.Series'> 
# imputer.fit(data['a'])
# data2 = imputer.transform(data['a'])
# print(data2)
# 에러가 뜬다 형태가 다르당

'''
[[ 2.        ]
 [ 6.66666667]
 [ 6.66666667]
 [ 8.        ]
 [10.        ]]
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   NaN  4.0   4.0  4.0
2   NaN  NaN   NaN  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''
# 이거 끼우는 방법 찾아보장
# data2 = imputer.fit_transform(data[['a']])
# data['a'] = 