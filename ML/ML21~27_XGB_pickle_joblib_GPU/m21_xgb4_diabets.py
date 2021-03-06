from tabnanny import verbose
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_boston,load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np,time,warnings
# warnings.filterwarnings(action='ignore')

### !!! QuantileTransformer,PowerTransformer,PolynomialFeatures가 뭐하는 기능인지 찾기

#1. 데이터
datasets = load_diabetes()   #,fetch_california_housing() load_boston()
x = datasets.data
y = datasets['target']
# print(x.shape,y.shape)  # (20640, 8) (20640,) (506,13)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)#, stratify=y

scaler = StandardScaler() 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = XGBRegressor()
model = XGBRegressor(
    n_jobs = -1,
    n_estimators = 2000,      # epochs느낌
    learning_rate = 0.25,
    max_depth = 4,          # 와 또 올라갔어    # tree깊이 몇개할거냐       LGBM은 비슷한 파라미터 하나 더 있다
    min_child_weight = 1,
    subsample=1,
    colsample_bytree = 1,
    reg_alpha = 0,          # 규제  L1       -> 둘 중에 하나만 할수도 있다.
    reg_lambda = 1,          # 규제  L2      -> 응용해서 나온개념 릿지와 랏소
)   # 온갖 파라미터들을 다 건드려보면서 값을 극한의 극한까지 끌어올릴수있다.
#3. 훈련

start = time.time()
model.fit(x_train,y_train)
end = time.time()

print(f'걸린시간 : {np.round(end - start,2)}초')

results = model.score(x_test,y_test)
print("result : ", np.round(results,4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print(f'r2 : {np.round(r2,4)}')

#   lr 0.075 
#    0.8587    