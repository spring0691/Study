import numpy as np, pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4,np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

#print(data.shape)   # (4,5)
data = data.transpose()
data.columns = ['a','b','c','d']
# print(data)
'''
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   NaN  4.0   4.0  4.0
2   NaN  NaN   NaN  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

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

data1 = data.fillna(means)
# print(data)
'''
           a         b          c    d
0   2.000000  2.000000   7.333333  6.0
1   6.666667  4.000000   4.000000  4.0
2   6.666667  4.666667   7.333333  6.0
3   8.000000  8.000000   8.000000  8.0
4  10.000000  4.666667  10.000000  6.0
'''

meds = data.median()    # 중위값    nan을 제외한 중위값
# print(meds)             # 짝수일 경우 중위2개값의 평균
'''
a    8.0
b    4.0
c    8.0
d    6.0
dtype: float64
'''
data2 = data.fillna(meds)
# print(data2)
'''
      a    b     c    d
0   2.0  2.0   8.0  6.0
1   8.0  4.0   4.0  4.0
2   8.0  4.0   8.0  6.0
3   8.0  8.0   8.0  8.0
4  10.0  4.0  10.0  6.0
'''

data2 = data.fillna(method='ffill')     # 앞의 값을 끌어다가 채워서 0행은 nan으로 뜬다.
# print(data2)                            # 전의 값 그대로 끌어와도 상관없는 데이터에서 씀.(시계열 등...)
'''
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   2.0  4.0   4.0  4.0
2   2.0  4.0   4.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0
'''

data2 = data.fillna(method='bfill')     # 뒤의 값을 끌어다가 채우서 제일 마지막행은 nan으로 뜬다.
# print(data2)
'''
      a    b     c    d
0   2.0  2.0   4.0  4.0
1   8.0  4.0   4.0  4.0
2   8.0  8.0   8.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

data2 = data.fillna(method='ffill', limit=1)    # 1개만 하겠다.
# print(data2)
'''
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   2.0  4.0   4.0  4.0
2   NaN  4.0   4.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0
'''
data2 = data.fillna(method='bfill', limit=1)  
# print(data2)
'''
      a    b     c    d
0   2.0  2.0   4.0  4.0
1   NaN  4.0   4.0  4.0
2   8.0  8.0   8.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

