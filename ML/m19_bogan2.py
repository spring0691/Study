import numpy as np, pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4,np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

#print(data.shape)   # (4,5)
data = data.transpose()
data.columns = ['a','b','c','d']
# print(data)

# 결측치 위치 확인.
#print(data.isnull())    # True로 표시되는 위치가 nan이다.
'''
       a      b      c      d
0  False  False   True   True
1   True  False  False  False
2   True   True   True   True
3  False  False  False  False
4  False   True  False   True
'''
# print(data.isnull().sum())
'''
a    2
b    2
c    2
d    3
dtype: int64
'''
# print(data.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   a       3 non-null      float64
 1   b       3 non-null      float64
 2   c       3 non-null      float64
 3   d       2 non-null      float64
dtypes: float64(4)
memory usage: 288.0 bytes
None
'''

#1. 결측치 삭제
# print(data.dropna())   # nan이 있는 행이 전부 삭제된다. axis=0은 행 axis=1은 열
# print(data.dropna(axis=0))    #   행삭제 default
# print(data.dropna(axis=1))    #   열삭제

#2. 특정값

means = data.mean()     # 평균값
# print(means)
'''
a    6.666667
b    4.666667
c    7.333333
d    6.000000
dtype: float64
'''

data = data.fillna(means)
# print(data)
'''
           a         b          c    d
0   2.000000  2.000000   7.333333  6.0
1   6.666667  4.000000   4.000000  4.0
2   6.666667  4.666667   7.333333  6.0
3   8.000000  8.000000   8.000000  8.0
4  10.000000  4.666667  10.000000  6.0
'''

