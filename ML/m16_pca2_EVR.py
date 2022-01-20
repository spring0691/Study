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

pca = PCA(n_components=14)
x = pca.fit_transform(x)    
print(x.shape)
# print(x[:5])

pca_EVR = pca.explained_variance_ratio_ # 설명가능한 변화율?
#print(pca_EVR)                          # 어떤 값들이 나온다. 이게 뭐지? 줄인 칼럼들에 대한 개수별 정확도?
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)             # 누적합을 구해준다.
print(cumsum)                           # 이런식으로 누적합을 보면서 몇개의 칼럼까지 빼도 성능이 괜찮은지 확인 할 수 있다.
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.        ]

import matplotlib.pyplot as plt
plt.plot(cumsum)
#plt.plot(pca_EVR)
plt.grid()      # 격자
plt.show()
'''
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
'''

