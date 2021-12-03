from tensorflow.keras.models import Sequential, Model         
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

# def RMSE(y_test, y_pred):
#     return np.sqrt(mean_squared_error(y_test,y_predict))    

#1. 데이터 로드 및 정제
path = "./_data/wine/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sample_Submission.csv')     

#일단 기본적으로 csv파일 직접보거나 excel로 열어서 colums값을 직접 보고 분석하는게 좋다.
print(train.info())

'''
x = train.drop(['datetime','casual','registered','count'], axis=1)  
test_file = test_file.drop(['datetime'], axis=1)

y = train['count']  

y = np.log1p(y) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)  

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
    
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    
test_file = scaler.transform(test_file)


#2. 모델링

input1 = Input(shape=(8,))
dense1 = Dense(16)(input1) #,activation="relu"
dense2 = Dense(24)(dense1)
dense3 = Dense(32)(dense2) #,activation="relu" 
dense4 = Dense(16)(dense3)
dense5 = Dense(8)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1,outputs=output1)
      
# model = Sequential()
# model.add(Dense(16, input_dim=8))    
# model.add(Dense(24)) #, activation='relu'
# model.add(Dense(32)) #, activation='relu'
# model.add(Dense(24)) 
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=5000,batch_size=50, verbose=1,validation_split=0.11111111,callbacks=[es])

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
# results = model.predict(test_file)

# submit_file['count'] = results  
# submit_file.to_csv(path + 'nolog_MaxAbs.csv', index=False)  
'''
'''                                 y값 로그O                                                   y값 로그X
결과정리                  일반레이어                      relu                    일반레이어                         relu
                                                                                    정확도고 자시고 -값때문에 다 리젝당함
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
