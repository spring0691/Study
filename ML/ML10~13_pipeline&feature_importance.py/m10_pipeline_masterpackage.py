from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,accuracy_score,f1_score
import numpy as np, pandas as pd, warnings, time
from sklearn.decomposition import PCA
warnings.filterwarnings(action='ignore')

scaler = MinMaxScaler()

path = '../Project/Kaggle_Project/bike/'
Bikedata = pd.read_csv(path + 'train.csv')                 

dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Bike':Bikedata,'Fetch_covtype':fetch_covtype()}

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
        model = make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier(random_state=66,n_jobs=-1))                                           
        start = time.time()
        model.fit(x_train,y_train)
        end = time.time()
        result = model.score(x_test,y_test)  
        print("model.score : ", result)
        print('\n')
        
    else:                    
        print('나는 회귀모델!')                                             # 회귀는 r2 ~ 
        model = make_pipeline(MinMaxScaler(),PCA(),RandomForestRegressor(random_state=66,n_jobs=-1))                                             
        start = time.time()
        model.fit(x_train,y_train)
        end = time.time()      
        result = model.score(x_test,y_test)   
        print("model.score : ", result)
        print('\n')

'''
Iirs 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  0.9


Breast_cancer 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  0.9736842105263158


Wine 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  1.0


Boston 데이터셋의 결과를 소개합니다~
나는 회귀모델!
model.score :  0.9266400209761564


Diabets 데이터셋의 결과를 소개합니다~
나는 회귀모델!
model.score :  0.36921653933532395


Bike 데이터셋의 결과를 소개합니다~
나는 회귀모델!
model.score :  0.2609731458217943


Fetch_covtype 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  0.9556207671058407

PCA() 넣고 난 후-----------------------------------------------------------
Iirs 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  0.9666666666666667


Breast_cancer 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  0.9385964912280702


Wine 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  0.9722222222222222


Boston 데이터셋의 결과를 소개합니다~
나는 회귀모델!
model.score :  0.8790489448743192


Diabets 데이터셋의 결과를 소개합니다~
나는 회귀모델!
model.score :  0.4284711709200252


Bike 데이터셋의 결과를 소개합니다~
나는 회귀모델!
model.score :  0.2998708954532694


Fetch_covtype 데이터셋의 결과를 소개합니다~
나는 분류모델!
model.score :  0.9547774153851449
'''
