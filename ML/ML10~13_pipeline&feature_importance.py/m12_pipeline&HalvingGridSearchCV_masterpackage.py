from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.experimental import enable_halving_search_cv   # halving이 개발중이라 사용하기 위해 import
from sklearn.model_selection import train_test_split,HalvingGridSearchCV,RandomizedSearchCV,GridSearchCV,HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor               # 분류용,회귀용 각각
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler # 오랜만에 보는 scaler 4총사
from sklearn.metrics import r2_score,accuracy_score,f1_score                            # 온갖 평가지표 
import numpy as np, pandas as pd, warnings, time, sys
from sklearn.decomposition import PCA 

# 출력 관련 옵션들
warnings.filterwarnings(action='ignore')
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',50)
pd.set_option('display.max_columns',50)

# 데이터 로드
path = '../Project/Kaggle_Project/bike/'
Bikedata = pd.read_csv(path + 'train.csv')                 
dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Bike':Bikedata,'Fetch_covtype':fetch_covtype()}

# 모델 설정들
n_splits = 5
parameters = {'Rfcl__n_estimators' : [100,200], 'Rfcl__max_depth' : [6, 8, 10, 12], 'Rfcl__min_samples_leaf' : [3, 5, 7, 10], 'Rfcl__min_samples_split' : [2, 3, 5, 10] }

cla_model = Pipeline([("mm",MinMaxScaler()),("PCA",PCA()),("Rfcl",RandomForestClassifier())])           # 분류 classifier
reg_model = Pipeline([("mm",MinMaxScaler()),("PCA",PCA()),("Rfcl",RandomForestRegressor())])            # 회귀 Regressor

classifier_model = HalvingGridSearchCV(cla_model, parameters, cv=n_splits, n_jobs=-1)      # 분류 classifier
regressor_model = HalvingGridSearchCV(reg_model, parameters, cv=n_splits, n_jobs=-1)       # 회귀 Regressor
# 이 검증에서는 K-fold 교차 검증 방법중 자동으로 Stratified K-fold 방법을 선택하여 분류 및 스코어링하는 것을 볼 수 있다.


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
Iirs 데이터셋의 결과를 소개합니다~
나는 분류모델!
최적의 매개변수는요~ :  Pipeline(steps=[('mm', MinMaxScaler()), ('PCA', PCA()),
                ('Rfcl',
                 RandomForestClassifier(max_depth=10, min_samples_leaf=5))])
최적의 파라미터는요~ :  {'Rfcl__max_depth': 10, 'Rfcl__min_samples_leaf': 5, 'Rfcl__min_samples_split': 2, 'Rfcl__n_estimators': 100}
model.score로 구한 값은요~ :  1.0
걸린 시간은요~ :  18.512936115264893
acc_score :  1.0
최적 튠 ACC :  1.0


Breast_cancer 데이터셋의 결과를 소개합니다~
나는 분류모델!
최적의 매개변수는요~ :  Pipeline(steps=[('mm', MinMaxScaler()), ('PCA', PCA()),
                ('Rfcl',
                 RandomForestClassifier(max_depth=8, min_samples_leaf=3,
                                        min_samples_split=3))])
최적의 파라미터는요~ :  {'Rfcl__max_depth': 8, 'Rfcl__min_samples_leaf': 3, 'Rfcl__min_samples_split': 3, 'Rfcl__n_estimators': 100}
model.score로 구한 값은요~ :  0.9298245614035088
걸린 시간은요~ :  20.516159534454346
acc_score :  0.9298245614035088
최적 튠 ACC :  0.9298245614035088


Wine 데이터셋의 결과를 소개합니다~
나는 분류모델!
최적의 매개변수는요~ :  Pipeline(steps=[('mm', MinMaxScaler()), ('PCA', PCA()),
                ('Rfcl',
                 RandomForestClassifier(max_depth=12, min_samples_leaf=5,
                                        n_estimators=200))])
최적의 파라미터는요~ :  {'Rfcl__max_depth': 12, 'Rfcl__min_samples_leaf': 5, 'Rfcl__min_samples_split': 2, 'Rfcl__n_estimators': 200}
model.score로 구한 값은요~ :  0.9722222222222222
걸린 시간은요~ :  21.20870876312256
acc_score :  0.9722222222222222
최적 튠 ACC :  0.9722222222222222

Boston 데이터셋의 결과를 소개합니다~
나는 회귀모델!
최적의 매개변수는요~ :  Pipeline(steps=[('mm', MinMaxScaler()), ('PCA', PCA()),
                ('Rfcl',
                 RandomForestRegressor(max_depth=8, min_samples_leaf=3,
                                       min_samples_split=3))])
최적의 파라미터는요~ :  {'Rfcl__max_depth': 8, 'Rfcl__min_samples_leaf': 3, 'Rfcl__min_samples_split': 3, 'Rfcl__n_estimators': 100}
model.score로 구한 값은요~ :  0.8455058825179809
걸린 시간은요~ :  20.605863094329834
r2_score :  0.8455058825179809
최적 튠 R2 :  0.8455058825179809


Diabets 데이터셋의 결과를 소개합니다~
나는 회귀모델!
최적의 매개변수는요~ :  Pipeline(steps=[('mm', MinMaxScaler()), ('PCA', PCA()),
                ('Rfcl',
                 RandomForestRegressor(max_depth=8, min_samples_leaf=5,
                                       min_samples_split=10))])
최적의 파라미터는요~ :  {'Rfcl__max_depth': 8, 'Rfcl__min_samples_leaf': 5, 'Rfcl__min_samples_split': 10, 'Rfcl__n_estimators': 100}
model.score로 구한 값은요~ :  0.44773165266961634
걸린 시간은요~ :  20.5342276096344
r2_score :  0.44773165266961634
최적 튠 R2 :  0.44773165266961634


Bike 데이터셋의 결과를 소개합니다~
나는 회귀모델!
최적의 매개변수는요~ :  Pipeline(steps=[('mm', MinMaxScaler()), ('PCA', PCA()),
                ('Rfcl',
                 RandomForestRegressor(max_depth=8, min_samples_leaf=5,
                                       min_samples_split=10))])
최적의 파라미터는요~ :  {'Rfcl__max_depth': 8, 'Rfcl__min_samples_leaf': 5, 'Rfcl__min_samples_split': 10, 'Rfcl__n_estimators': 100}
model.score로 구한 값은요~ :  0.34441515307436166
걸린 시간은요~ :  33.54323196411133
r2_score :  0.34441515307436166
최적 튠 R2 :  0.34441515307436166
'''