import numpy as np, pandas as pd, warnings, time
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes,fetch_california_housing
from sklearn.model_selection import train_test_split            
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings(action="ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    #이게 LDA   
from sklearn.decomposition import PCA   # decomposition 분해
from xgboost import XGBRegressor,XGBClassifier


path = '../Project/Kaggle_Project/bike/'
Bikedata = pd.read_csv(path + 'train.csv') 
dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'California':fetch_california_housing(),'Boston':load_boston(),'Diabets':load_diabetes(),'Bike':Bikedata,'Fetch_covtype':fetch_covtype()}


for name,data in dd.items():
    
    if name == 'Bike':
        x = Bikedata.drop(['casual','registered','count'], axis=1)  
        x['datetime'] = pd.to_datetime(x['datetime'])
        x['year'] = x['datetime'].dt.year
        x['month'] = x['datetime'].dt.month
        x['day'] = x['datetime'].dt.day
        x['hour'] = x['datetime'].dt.hour
        x = x.drop('datetime', axis=1)
        y = Bikedata['count']  # np.unique(y, return_counts = True 누가봐도 회귀모델
        y = np.log1p(y)
    else:
        datasets = data
        x = datasets.data
        y = datasets.target
        #print(datasets.feature_names)
    
    choice = np.unique(y, return_counts=True)[1].min()    # 회귀모델인지 분류모델인지 판별해주는 변수 y값 라벨들의 개수 중 제일 적은 개수를 보고판단
    print(f'{name} 데이터셋의 결과를 소개합니다~\n')
    
    if choice > 4:
        print('나는 분류모델! stratify 해줘요~')
        x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66, stratify=y)    # stratify <-- 나눌때 편향 안되게 막아준다.
    else:
        print('나는 회귀모델! stratify 안해요~')
        x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
    print('LDA 전 : ',x_train.shape)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if choice < 4 :
        print('회귀모델은 LDA하기 위해 y_train을 int로 바꿔줘야합니다.')   # 다양한 방법이 있지요
        y_train = np.round(y_train)
        
    lda = LinearDiscriminantAnalysis()   
    x_train = lda.fit_transform(x_train, y_train)         # y를 보고 한다는게 어마어마한거다. 어마어마한 압축률과 성능을 자랑한다.
    x_test = lda.transform(x_test)
    print('LDA 후 : ',x_train.shape)                      # default값은 y_classes - 1 값이다.
    
    # eval_metric='error'  eval_metric='merror'    
    # eval_metric는 loss 같은 개념이다. rmse,rmsle,mae,mape,logloss,error,meroror 등이있다. error은 2진분류 merror은 다중분류, 그외에 logloss등은 회귀. 훈련에 영향을 끼친다.
    
    if choice > 4:       
        print('나는 분류모델!')
        m = len(np.unique(y))    # 이진분류인지 다중분류인지 판별해주는 변수
        if m == 2:                        
            model = XGBClassifier(eval_metric='error')           
        else : 
            model = XGBClassifier(eval_metric='merror')                              
        start = time.time()
        model.fit(x_train,y_train)
        end = time.time()
        
        print('model.score로 구한 값은요~ : ', model.score(x_test,y_test))   
        print('걸린 시간은요~ : ', end - start)
        
        print('\n')
        
    else:                    
        print('나는 회귀모델!')                                              
        model = XGBRegressor()                                     
        start = time.time()
        model.fit(x_train,y_train)
        end = time.time()      
       
        print('model.score로 구한 값은요~ : ', model.score(x_test,y_test))   
        print('걸린 시간은요~ : ', end - start)
                      
        print('\n')
    
'''
Iirs 데이터셋의 결과를 소개합니다~

나는 분류모델! stratify 해줘요~
LDA 전 :  (120, 4)
LDA 후 :  (120, 2)
나는 분류모델!
model.score로 구한 값은요~ :  1.0
걸린 시간은요~ :  0.08851790428161621


Breast_cancer 데이터셋의 결과를 소개합니다~

나는 분류모델! stratify 해줘요~
LDA 전 :  (455, 30)
LDA 후 :  (455, 1)
나는 분류모델!
model.score로 구한 값은요~ :  0.9473684210526315
걸린 시간은요~ :  0.042015790939331055


Wine 데이터셋의 결과를 소개합니다~

나는 분류모델! stratify 해줘요~
LDA 전 :  (142, 13)
LDA 후 :  (142, 2)
나는 분류모델!
model.score로 구한 값은요~ :  1.0
걸린 시간은요~ :  0.06927919387817383


California 데이터셋의 결과를 소개합니다~

나는 회귀모델! stratify 안해요~
LDA 전 :  (16512, 8)
회귀모델은 LDA하기 위해 y_train을 int로 바꿔줘야합니다.
LDA 후 :  (16512, 5)
나는 회귀모델!
model.score로 구한 값은요~ :  0.6968932596722985
걸린 시간은요~ :  0.9419434070587158


Boston 데이터셋의 결과를 소개합니다~

나는 회귀모델! stratify 안해요~
LDA 전 :  (404, 13)
회귀모델은 LDA하기 위해 y_train을 int로 바꿔줘야합니다.
LDA 후 :  (404, 13)
나는 회귀모델!
model.score로 구한 값은요~ :  0.9020750951190113
걸린 시간은요~ :  0.11534428596496582


Diabets 데이터셋의 결과를 소개합니다~

나는 회귀모델! stratify 안해요~
LDA 전 :  (353, 10)
회귀모델은 LDA하기 위해 y_train을 int로 바꿔줘야합니다.
LDA 후 :  (353, 10)
나는 회귀모델!
model.score로 구한 값은요~ :  0.313354229055848
걸린 시간은요~ :  0.11080241203308105


Bike 데이터셋의 결과를 소개합니다~

나는 회귀모델! stratify 안해요~
LDA 전 :  (8708, 12)
회귀모델은 LDA하기 위해 y_train을 int로 바꿔줘야합니다.
LDA 후 :  (8708, 6)
나는 회귀모델!
model.score로 구한 값은요~ :  0.8189430469073561
걸린 시간은요~ :  0.5407247543334961


Fetch_covtype 데이터셋의 결과를 소개합니다~

나는 분류모델! stratify 해줘요~
LDA 전 :  (464809, 54)
LDA 후 :  (464809, 6)
나는 분류모델!
model.score로 구한 값은요~ :  0.7878109859470065
걸린 시간은요~ :  153.06962537765503
'''