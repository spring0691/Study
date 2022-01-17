from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

scaler = MinMaxScaler()

n_splits = 4
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)             
Skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66) 

dd =  {'Boston':load_boston(),'Diabets':load_diabetes(),'Fetch_covtype':fetch_covtype()}#'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),

parameters = [{'n_estimators' : [100,200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10], 
               'min_samples_split' : [2, 3, 5, 10] }]    # , 'n_jobs : ' : [-1, 2, 4, 6]

#print(RandomForestClassifier().get_params().keys())   # estimator의 파라미터 값들을 확인할 수 있다.


regressor_model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, n_jobs=-1)       # 회귀모델   회귀모델은 Regressor 써주고
classifier_model = GridSearchCV(RandomForestClassifier(), parameters, cv=Skfold, n_jobs=-1)     # 분류모델  분류모델은 classifier써줘야한다.

for name,data in dd.items():
    datasets = data
    x = datasets.data
    y = datasets.target
    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    choice = len(np.unique(y))
    
    print(f'{name} 데이터셋의 결과를 소개합니다~')
    
    if choice <= 10:        
        model = classifier_model
        model.fit(x_train,y_train)
        print('나는 분류모델!')
        print(model.score(x_test,y_test))
        print('\n')
        
    elif choice > 10:
        model = regressor_model      
        model.fit(x_train,y_train)
        print('나는 회귀모델!')      
        print(model.score(x_test,y_test))
        print('\n')

'''
Iirs 데이터셋의 결과를 소개합니다~
나는 분류모델!
1.0


Breast_cancer 데이터셋의 결과를 소개합니다~
나는 분류모델!
0.9649122807017544


Wine 데이터셋의 결과를 소개합니다~
나는 분류모델!
1.0

Boston 데이터셋의 결과를 소개합니다~
나는 회귀모델!
0.9222251784295408


Diabets 데이터셋의 결과를 소개합니다~
나는 회귀모델!
0.38714743139195007
'''