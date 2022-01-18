from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,accuracy_score,f1_score
import numpy as np, pandas as pd, warnings, time

warnings.filterwarnings('always')


#scaler = MinMaxScaler()

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)             
Skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66) 


parameters = {'n_estimators' : [100,200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [2, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10] }

regressor_model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv=kfold, n_jobs=-1)        # 회귀 Regressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)


model = regressor_model                                             
start = time.time()
model.fit(x_train,y_train)
end = time.time()      
print("최적의 매개변수는요~ : ", model.best_estimator_)
print('최적의 파라미터는요~ : ', model.best_params_)
print('model.score로 구한 값은요~ : ', model.score(x_test,y_test))   
print('걸린 시간은요~ : ', end - start)

y_pred = model.predict(x_test)
print("r2_score : ", r2_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)                
print("최적 튠 R2 : ", r2_score(y_test,y_pred_best))              
print('\n')

