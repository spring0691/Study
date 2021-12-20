from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))    #배열의 제곱근을 계산 np.sqrt

#1. 데이터 로드 및 정제
path = "../_data/kaggle/bike/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                   
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     

x = train.drop(['datetime','casual','registered','count'], axis=1)  
test_file = test_file.drop(['datetime'], axis=1)

y = train['count'] 

#y = np.log1p(y) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)  

#2. 모델링      
model = Sequential()
model.add(Dense(128, input_dim=8))    
model.add(Dense(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') #kaggle에서 이 모델은 RMSLE로 loss값 잡으라고 명시해줌

es = EarlyStopping(monitor = "val_loss", patience=5, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=1000,batch_size=10, verbose=1,validation_split=0.25,callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)   
   
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("R2 : ", r2)

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

print(y_predict[71:81])
print(y_test[71:81])
print(y_test.shape)
print(type(y_test))
############################# 제출용 제작 ####################################
results = model.predict(test_file)

submit_file['count'] = results  #test_file에서 예측한 값을 results에 담아서 그 값을 submit_file의 count열에 넣어준다.

print(submit_file[:10]) #확인해보면 count 값이 0에서 model.predict(test_file)에서 나온 결과값이 다 들어가져있는걸 확인 할 수 있다.

submit_file.to_csv(path + '12.16_test.csv', index=False) # index를 넣지않겠다는 설정    
