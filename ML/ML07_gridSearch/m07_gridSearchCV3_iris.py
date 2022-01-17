from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,accuracy_score,f1_score
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

scaler = MinMaxScaler()

n_splits = 4
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)             
Skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66) 

dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Fetch_covtype':fetch_covtype()}

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
        model = classifier_model                                           # 분류는 acc, 혹시 모르니까 f1~
        model.fit(x_train,y_train)
        print('나는 분류모델!')
        print("최적의 매개변수는요~ : ", model.best_estimator_)
        print('최적의 파라미터는요~ : ', model.best_params_)
        print('model.score로 구한 값은요~ : ', model.score(x_test,y_test))   # GridSearch가 최적의 값을 찾아줬다면 이게 제일 높은 score
        
        y_pred = model.predict(x_test)                                      # 실전에서 x_test를 predict해보고 
        print("acc_score : ", accuracy_score(y_test,y_pred))                # 거기서 나온 acc를 기록 model.socre값과 이 아래의 값과 비교해본다. 검증의 검증.
        #print("f1_score : ", f1_score(y_test,y_pred))                       # f1 score로 편향되었는지 확인
        
        y_pred_best = model.best_estimator_.predict(x_test)                # best_estimator로 predict 해본다. 가장 좋은값!
        print("최적 튠 ACC : ", accuracy_score(y_test,y_pred_best))        # 여기서 acc 확인함으로써 best값이 model.score로 전달 되었나 안 되었나 확인할수있다.
        #print("최적 튠 F1 : ", f1_score(y_test,y_pred_best))
        print('\n')
        
    elif choice > 10:                                                      
        model = regressor_model                                             # 회귀는 r2 ~ 
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
Iirs 데이터셋의 결과를 소개합니다~
나는 분류모델!
최적의 매개변수는요~ :  RandomForestClassifier(max_depth=6, min_samples_leaf=7, min_samples_split=3)
최적의 파라미터는요~ :  {'max_depth': 6, 'min_samples_leaf': 7, 'min_samples_split': 3, 'n_estimators': 100}
model.score로 구한 값은요~ :  1.0
acc_score :  1.0
최적 튠 ACC :  1.0


Breast_cancer 데이터셋의 결과를 소개합니다~
나는 분류모델!
최적의 매개변수는요~ :  RandomForestClassifier(max_depth=12, min_samples_leaf=3)
최적의 파라미터는요~ :  {'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}
model.score로 구한 값은요~ :  0.9649122807017544
acc_score :  0.9649122807017544
최적 튠 ACC :  0.9649122807017544


Wine 데이터셋의 결과를 소개합니다~
나는 분류모델!
최적의 매개변수는요~ :  RandomForestClassifier(max_depth=10, min_samples_leaf=3, min_samples_split=5,
                       n_estimators=200)
최적의 파라미터는요~ :  {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 200}
model.score로 구한 값은요~ :  1.0
acc_score :  1.0
최적 튠 ACC :  1.0


Boston 데이터셋의 결과를 소개합니다~
나는 회귀모델!
최적의 매개변수는요~ :  RandomForestRegressor(max_depth=10, min_samples_leaf=3, min_samples_split=3,
                      n_estimators=200)
최적의 파라미터는요~ :  {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 200}
model.score로 구한 값은요~ :  0.9198879775751767
r2_score :  0.9198879775751767
최적 튠 R2 :  0.9198879775751767


Diabets 데이터셋의 결과를 소개합니다~
나는 회귀모델!
최적의 매개변수는요~ :  RandomForestRegressor(max_depth=8, min_samples_leaf=7, min_samples_split=3)
최적의 파라미터는요~ :  {'max_depth': 8, 'min_samples_leaf': 7, 'min_samples_split': 3, 'n_estimators': 100}
model.score로 구한 값은요~ :  0.39491478167108496
r2_score :  0.39491478167108496
최적 튠 R2 :  0.39491478167108496
'''