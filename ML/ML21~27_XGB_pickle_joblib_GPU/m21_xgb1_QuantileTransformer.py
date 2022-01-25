from tabnanny import verbose
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

### !!! QuantileTransformer,PowerTransformer,PolynomialFeatures가 뭐하는 기능인지 찾기

#1. 데이터
datasets = load_boston()   #,fetch_california_housing()
x = datasets.data
y = datasets['target']
# print(x.shape,y.shape)  # (20640, 8) (20640,)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)#, stratify=y

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = XGBRegressor()
model = XGBRegressor(
    n_jobs = -1,
    n_estimators = 200,      # epochs느낌
    learning_rate = 0.05
)
#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train,verbose=2)
end = time.time()

print(f'걸린시간 : {np.round(end - start,4)}')

results = model.score(x_test,y_test)
print("result : ", np.round(results,4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print(f'r2 : {np.round(r2,4)}')

# california
#       default         200       200 + rl 0.05     200 + rl 0.1    n_est2000 + njobs -1 + rl0.1
#r2      0.8433       0.8444        0.8459            0.8532                0.8566

# boston
#                                                                   n_est2000 + njobs -1 + rl0.1
#r2                                                                         0.9313
# 각종 스케일러 변경 및 learing_rate설정으로 값을 최대로 올려보자           0.9339 (robust)

print('------------------------------------')
# hist = model.evals_result()
# print(hist)