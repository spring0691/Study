import numpy as np, pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer
import warnings

warnings.filterwarnings(action='ignore')
#1. 데이터 로드

datasets = load_iris()
# print(datasets)               # 그냥 내용 다 출력
#print(datasets.feature_names)   # 칼럼명 출력
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data       # 현재 numpy 형태
y = datasets.target   # 현재 numpy 형태
 
x = pd.DataFrame(x, columns=datasets['feature_names'])
# x = pd.DataFrame(x, columns=datasets.feature_names)
#x = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])      3개가 다 같은 의미
x['Target'] = y     # y값들을 넣어줄건데 그 column의 이름을 Target로 지어서 추가하겠다.
#print(x[:5])   #깔끔하게 정리되어서 출력되는 걸 확인 할 수 있다.

print('=========================상관계수 히트 맵 ======================')
print(x.corr())                 # 전에 했다. 각 칼럼들의 서로의 대한 상관관계가 나온다.

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=x.corr(), square=True, annot=True, cbar=True)
# cbar = 그림 옆에 나오는 색깔의 수치를 나타내 주는 표
# 상관관계에서 주는 수치는 서로서로의 colmun에 대한 linear관계
# 여기서 linear한 관계란 직관적인 상관관계를 의미한다 각 칼럼에 대한 y = wx + b
# 더 직관적으로 풀어보자면 x1 = wx2 + b , x1 = wx3 + b, x1 = wx4 + b 이런식으로 
# 모든 칼럼에 대한 grid 격자치기로 값을 나타내주기때문에 서로서로의 상관관계를 한번에 알 수 있다.
# 이 상관관계로 서로 수치가 높은 칼럼들끼리 PCA로 묶어서 칼럼을 줄여주는 식으로 활용할 수 있다. 
plt.show()
