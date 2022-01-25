# 결측치 처리
#1. 행또는 열 삭제
#2. 임의의 값
#   fillna - 0, frontfill, backfill, 중위값, 평균값
#3. 보간 - interpolate
#4. 모델링 - predict
#5. 부스팅계열 - 통상 결측치, 이상치에 대해 자유롭다.
#감성적인 결측치,이상치 예측도 매우 중요하다. 데이터에 대한 자세한 분석이 필요하다.
#ex)주가,기상,체온 등등 상식적인 부분에서의 결측치는 평균치로 내는것보다 데이터에 대한 이해가 훨씬 중요하다
# 7~8월에 장마로 인한 강수량 증가, 체온은 33~37도가 상식적인 범위. 

import pandas as pd, numpy as np
from datetime import datetime

dates = ['1/24/2022','1/25/2022','1/26/2022','1/27/2022','1/28/2022']

dates = pd.to_datetime(dates)
#print(dates)    #DatetimeIndex(['2022-01-24', '2022-01-25', '2022-01-26', '2022-01-27', '2022-01-28'],dtype='datetime64[ns]', freq=None)

# Series는 vector dataframe은 행렬개념
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates)
#print(ts)
'''
2022-01-24     2.0
2022-01-25     NaN
2022-01-26     NaN
2022-01-27     8.0
2022-01-28    10.0
dtype: float64 
'''

ts = ts.interpolate()   # 보간법.
print(ts)
'''
2022-01-24     2.0
2022-01-25     4.0
2022-01-26     6.0
2022-01-27     8.0
2022-01-28    10.0
dtype: float64
'''