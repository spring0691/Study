from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target

'''
print(x)    # x내용물 확인
print(y)    # y내용물 확인
print(x.shape) # x형태  (506,13)    -> (506,13,1,1) 해서 cnn으로 모델링
print(y.shape) # y형태  (506, )
print(datasets.feature_names) # 컬럼,열의 이름들
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
'''

# cnn 만들기

# numpy pandas로 변환후 pandas의 제공기능인 index정보와 columns정보를 확인할수있다.
xx = pd.DataFrame(x, columns=datasets.feature_names)    # x가 pandas로 바껴서 xx에 저장, columns를 칼럼명이 나오게 지정해준다.
#print(type(xx))         # pandas.core.frame.DataFrame
#print(xx)               # 확인

#print(xx.corr())        # 칼럼들의 서로서로의 상관관게를 확인할 수 있다.    절대값클수록 양 or 음의 상관관계 0에 가까울수록 서로 영향 없음

xx['price'] = y         # xx의 데이터셋에 y값을 price라는 이름의 칼럼으로 추가한다. 원본데이터는 그대로있다.

print(xx)

print(xx.corr())

import matplotlib.pyplot as plt
import seaborn as sns   # 조금 더 이쁘게 만들게 도와줌.

plt.figure(figsize=(10,10))
sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# seaborn heatmap 개념정리

plt.show()
