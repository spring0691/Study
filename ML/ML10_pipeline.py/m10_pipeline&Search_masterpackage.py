from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,accuracy_score,f1_score
import numpy as np, pandas as pd, warnings, time

warnings.filterwarnings(action='ignore')

scaler = MinMaxScaler()

path = '../Project/Kaggle_Project/bike/'
Bikedata = pd.read_csv(path + 'train.csv')                 

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)                        # 회귀 Regressor
Skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)             # 분류 calssifier    편향막기위해(한쪽 데이터셋의 비율이 큼) Stratified

dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Bike':Bikedata,'Fetch_covtype':fetch_covtype()}

parameters = {'n_estimators' : [100,200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10] }

reg_model = make_pipeline(MinMaxScaler(),RandomForestRegressor())
cla_model = make_pipeline(MinMaxScaler(),RandomForestClassifier())

regressor_model = HalvingGridSearchCV(reg_model, parameters, cv=kfold, n_jobs=-1)       # 회귀 Regressor
classifier_model = HalvingGridSearchCV(cla_model, parameters, cv=Skfold, n_jobs=-1)     # 분류 classifier

for name,data in dd.items():
    
    if name == 'Bike':
        x = Bikedata.drop(['datetime','casual','registered','count'], axis=1)  
        y = Bikedata['count']  # np.unique(y, return_counts = True 누가봐도 회귀모델
    else:
        datasets = data
        x = datasets.data
        y = datasets.target
    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
    choice = np.unique(y, return_counts=True)[1].min()    # 회귀모델인지 분류모델인지 판별해주는 변수 y값 라벨들의 개수 중 제일 적은 개수를 보고판단
    
    print(f'{name} 데이터셋의 결과를 소개합니다~')
    
    if choice > 4:       
        print('나는 분류모델!')                                             # 분류는 acc, 혹시 모르니까 f1~
        model = classifier_model                                           
        start = time.time()
        model.fit(x_train,y_train)
        end = time.time()
        print("최적의 매개변수는요~ : ", model.best_estimator_)
        print('최적의 파라미터는요~ : ', model.best_params_)
        print('model.score로 구한 값은요~ : ', model.score(x_test,y_test))   # GridSearch가 최적의 값을 찾아줬다면 이게 제일 높은 score
        print('걸린 시간은요~ : ', end - start)
        
        y_pred = model.predict(x_test)                                      # 실전에서 x_test를 predict해보고 
        print("acc_score : ", accuracy_score(y_test,y_pred))                # 거기서 나온 acc를 기록 model.socre값과 이 아래의 값과 비교해본다. 검증의 검증.
        #print("f1_score : ", f1_score(y_test,y_pred))                       # f1 score로 편향되었는지 확인
        
        y_pred_best = model.best_estimator_.predict(x_test)                # best_estimator로 predict 해본다. 가장 좋은값!
        print("최적 튠 ACC : ", accuracy_score(y_test,y_pred_best))        # 여기서 acc 확인함으로써 best값이 model.score로 전달 되었나 안 되었나 확인할수있다.
        #print("최적 튠 F1 : ", f1_score(y_test,y_pred_best))
        print('\n')
        
    else:                    
        print('나는 회귀모델!')                                             # 회귀는 r2 ~ 
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
        
'''

'''