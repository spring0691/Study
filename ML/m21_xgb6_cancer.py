from tabnanny import verbose
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_boston,load_wine,load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
import numpy as np,time,warnings
# warnings.filterwarnings(action='ignore')

### !!! QuantileTransformer,PowerTransformer,PolynomialFeatures가 뭐하는 기능인지 찾기

#1. 데이터
datasets = load_breast_cancer()   #,fetch_california_housing() load_boston()
x = datasets.data
y = datasets['target']
# print(x.shape,y.shape)  # (20640, 8) (20640,) (506,13)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)#, stratify=y

scaler = MinMaxScaler() #RobustScaler   MinMaxScaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# vervose가 보고싶은데 안나온다. 어떻게 해야할까? -> 어떤걸로 평가할건지 평가 지표를 설정해줘야 vervose가 나온다.
# model = XGBRegressor()
model = XGBClassifier(
    n_jobs = -1,
    n_estimators = 2000,      # epochs느낌
    learning_rate = 0.075,
    max_depth = 5,          # 와 또 올라갔어    # tree깊이 몇개할거냐       LGBM은 비슷한 파라미터 하나 더 있다
    min_child_weight = 1,
    subsample=1,
    colsample_bytree = 1,
    reg_alpha = 0,          # 규제  L1       -> 둘 중에 하나만 할수도 있다.
    reg_lambda = 1,          # 규제  L2      -> 응용해서 나온개념 릿지와 랏소
)   # 온갖 파라미터들을 다 건드려보면서 값을 극한의 극한까지 끌어올릴수있다.

#3. 훈련

start = time.time()
model.fit(x_train,y_train,verbose=1,eval_set=[(x_test,y_test)])
end = time.time()

print(f'걸린시간 : {np.round(end - start,2)}초')

results = model.score(x_test,y_test)
print("result : ", np.round(results,4))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print(f'acc : {np.round(acc,4)}')

# 다중분류는 mlogloss. 이진분류는 logloss. 회귀는 rmse