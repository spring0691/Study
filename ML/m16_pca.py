# 주성분 분석(主成分分析, Principal component analysis; PCA)은 고차원의 데이터를 저차원의 데이터로 환원시키는 기법을 말한다. -> 차원 축소
# 모든 columns,feature가 결과값에 영향을 미치지는 않는다. 예전에 embedding에서 vector화 한것도 비슷한 개념이다.
# 데이터도 embedding처럼 압축해서 사이즈를 줄여보자?  2차원 데이터를 1차원으로 만드는데 2차원 사이의 점들의 거리를 1차원에서 점들의 거리로 표현하여 차원을 줄인다.
# scalering이랑 비슷한 개념. y값은 그대로 있는 상태에서 x값만 그 차원을,범위를,수치를 줄여준다.

import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA   # decomposition 분해
from sklearn.model_selection import train_test_split

#1. 데이터로드 및 정제

# datasets = load_boston()
#datasets = fetch_california_housing()
datasets = load_breast_cancer()
x = datasets.data       # (506,13)
y = datasets.target
print(x.shape)
#print(datasets.feature_names)

# from sklearn.datasets import fetch_openml
# housing = fetch_openml(name="house_prices", as_frame=True)

pca = PCA(n_components=13)   # 아무조건 없이 columns을 줄여주기때문에 비지도학습과 유사하다 -> 압축했다가 다시 되돌리면 자잘한 데이터 날릴수있다 마치 log1p -> exmp1하는거처럼
x = pca.fit_transform(x)    
print(x.shape)
# print(x[:5])

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=66, shuffle=True)

#2. 모델
from xgboost import XGBRegressor,XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='error')
# model = XGBRegressor

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
results = model.score(x_test,y_test)
print('결과 : ', results)

# import sklearn as sk
# print(sk.__version__)
