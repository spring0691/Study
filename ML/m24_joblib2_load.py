from tabnanny import verbose
import joblib
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np,time,warnings
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')

### !!! QuantileTransformer,PowerTransformer,PolynomialFeatures가 뭐하는 기능인지 찾기

#1. 데이터
datasets = load_boston()   #,fetch_california_housing() load_boston()
x = datasets.data
y = datasets['target']
# print(x.shape,y.shape)  # (20640, 8) (20640,) (506,13)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)#, stratify=y

scaler = MinMaxScaler() #RobustScaler   MinMaxScaler
# x_train = scaler.fit_transform(x_train)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델 불러오기   // 2.모델,3.훈련(weight)
# import pickle
path = 'D:\_data\_save/'
# model = pickle.load(open(path+'m23_pickle1_save','rb')) # 지금은 read binary 중요!
model = joblib.load(path + 'm24_joblib1_save.dat')

#4. 평가
results = model.score(x_test,y_test)
print(f'results : {results}\n')

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print(f'r2 : {r2}\n')

hist = model.evals_result()
print(f'hist_0 : {hist.get("validation_0")}\n')
print(f'hist_1 : {hist.get("validation_1")}')