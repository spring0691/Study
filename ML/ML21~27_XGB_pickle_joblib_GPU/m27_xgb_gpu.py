from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_boston,fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np,time,warnings
import matplotlib.pyplot as plt
# warnings.filterwarnings(action='ignore')

#1. 데이터
# datasets = fetch_covtype()   #,fetch_california_housing() load_boston()
# x = datasets.data
# y = datasets['target']
# print(x.shape,y.shape)  # (581012, 54) (581012,)

import pickle
path = 'D:\_data\_save/'
datasets = pickle.load(open(path + 'm23_pickle1_save_datasets.dat','rb'))
x = datasets.data
y = datasets['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)#, stratify=y


scaler = MinMaxScaler() #RobustScaler   MinMaxScaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 저장
# pickle.dump(datasets, open(path + 'm23_pickle1_save_datasets.dat','wb',))  # write binary
# model = pickle.load(open(path+'m23_pickle1_save','rb'))

#2. 모델
# vervose가 보고싶은데 안나온다. 어떻게 해야할까? -> 어떤걸로 평가할건지 평가 지표를 설정해줘야 vervose가 나온다.
# model = XGBRegressor()
model = XGBClassifier(
    n_jobs = -1,
    n_estimators = 100,      # epochs느낌
    learning_rate = 0.075,
    max_depth = 5,          # 와 또 올라갔어    # tree깊이 몇개할거냐       LGBM은 비슷한 파라미터 하나 더 있다
    min_child_weight = 1,
    subsample=1,
    colsample_bytree = 1,
    reg_alpha = 0,          # 규제  L1       -> 둘 중에 하나만 할수도 있다.
    reg_lambda = 1,          # 규제  L2      -> 응용해서 나온개념 릿지와 랏소   가중치 규제하는것.
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor'
)   # 온갖 파라미터들을 다 건드려보면서 값을 극한의 극한까지 끌어올릴수있다.

#3. 훈련

start = time.time()
model.fit(x_train,y_train,verbose=1,eval_set=[(x_train,y_train),(x_test,y_test)],eval_metric='merror',
         early_stopping_rounds=50) # loss기준으로 관측한다.  early_stopping_rounds=10
# eval_metric -> loss같은 개념      # rmse, mae, logloss, error, merror, 
end = time.time()

print(f'걸린시간 : {np.round(end - start,2)}초')

results = model.score(x_test,y_test)
print("result : ", np.round(results,4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print(f'r2 : {np.round(r2,4)}')

'''
CPU연산
걸린시간 : 80.09초
result :  0.7802
r2 : 0.4655

GPU연산
걸린시간 : 5.95초
result :  0.7801
r2 : 0.4617 
'''

