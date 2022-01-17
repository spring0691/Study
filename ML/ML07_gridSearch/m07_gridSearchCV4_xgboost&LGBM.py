import pandas as pd, os, numpy as np
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings(action='ignore')

path = '../Project/Kaggle_Project/bike/'

train = pd.read_csv(path + 'train.csv')                 

x = train.drop(['datetime','casual','registered','count'], axis=1)  
y = train['count']  # np.unique(y, return_counts = True 누가봐도 회귀모델
 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
'''
parameters = [{'n_estimators' : [100,200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10], 
               'min_samples_split' : [2, 3, 5, 10] }]

model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, n_jobs=-1) 

parameters = {'n_estimators':[100, 500],'learning_rate':[0.05,0.1]}
model = GridSearchCV(GradientBoostingRegressor(), param_grid=parameters, cv=kfold, n_jobs=-1)
'''
parameters = {'max_depth' : [3],
          'eta': [0.1],
          'objective':['reg:linear'],
          'eval_metric':['logloss'],
          'early_stopping':[100]
          }
model = GridSearchCV(XGBRegressor(), parameters, cv=kfold, n_jobs=-1)
#model = GridSearchCV(LGBMRegressor(), parameters, cv=kfold, n_jobs=-1)
#model = GridSearchCV(CatBoostRegressor(),  cv=kfold, n_jobs=-1)

model.fit(x_train,y_train)
print('나는 회귀모델!')      
print("최적의 매개변수는요~ : ", model.best_estimator_)
print('최적의 파라미터는요~ : ', model.best_params_)
print('model.score로 구한 값은요~ : ', model.score(x_test,y_test))   

y_pred = model.predict(x_test)
print("r2_score : ", r2_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)                
print("최적 튠 R2 : ", r2_score(y_test,y_pred_best))              
print('\n')

'''
나는 회귀모델!
최적의 매개변수는요~ :  RandomForestRegressor(max_depth=10, min_samples_leaf=3, min_samples_split=5,
                      n_estimators=200)
최적의 파라미터는요~ :  {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 200}
model.score로 구한 값은요~ :  0.36442296420636966
r2_score :  0.36442296420636966
최적 튠 R2 :  0.36442296420636966

나는 회귀모델!
최적의 매개변수는요~ :  GradientBoostingRegressor(n_estimators=500)
최적의 파라미터는요~ :  {'learning_rate': 0.1, 'n_estimators': 500}
model.score로 구한 값은요~ :  0.3402718133230821
r2_score :  0.3402718133230821
최적 튠 R2 :  0.3402718133230821


'''