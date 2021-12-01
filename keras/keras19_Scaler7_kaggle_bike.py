from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))    #배열의 제곱근을 계산 np.sqrt

#1. 데이터 로드 및 정제
path = "./_data/bike/"   

train = pd.read_csv(path + 'train.csv')                 #print(train)    #[10886 rows x 12 columns]
test_file = pd.read_csv(path + 'test.csv')                   #print(test)     #[6493 rows x 9 columns]
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     #print(submit)    #[6493 rows x 2 columns]

x = train.drop(['datetime','casual','registered','count'], axis=1)  
test_file = test_file.drop(['datetime'], axis=1)

y = train['count']  

y = np.log1p(y) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)  

#scaler = MinMaxScaler()   #어떤 스케일러 사용할건지 정의부터 해준다.
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
    
scaler.fit(x_train)       #어떤 비율로 변환할지 계산해줌.
x_train = scaler.transform(x_train)   # 훈련할 데이터 변환
x_test = scaler.transform(x_test)    # test할 데이터도 비율로 변환. 설령 -값이거나 1을 초과해도 이미 weight구했으므로 예측값이나온다.
test_file = scaler.transform(test_file)


#2. 모델링      
model = Sequential()
model.add(Dense(50, input_dim=8))    
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=5000,batch_size=100, verbose=1,validation_split=0.25,callbacks=[es])

#4. 평가
loss = model.evaluate(x_test,y_test)   
print('loss값 : ', loss) 

#5. 예측     비교해보면 정확도를 검증할수 있다
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("R2 : ", r2)

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)


############################# 제출용 제작 ####################################
results = model.predict(test_file)

submit_file['count'] = results  
submit_file.to_csv(path + 'nomarl layer test.csv', index=False)  

'''
결과정리                일반레이어          relu

안하고 한 결과 
loss값 :  
R2 :  
RMSE : 

MinMax
loss값 :  
R2 : 
RMSE : 

Standard
loss값 : 
R2 :  
RMSE :  

Robust
loss값 : 
R2 :  
RMSE :  

MaxAbs
loss값 : 
R2 :  
RMSE :  
'''