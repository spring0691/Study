from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
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

#y = np.log1p(y) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)  

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
model.add(Dense(16, input_dim=8))    
model.add(Dense(24, activation='relu')) #
model.add(Dense(32, activation='relu')) #
model.add(Dense(24)) 
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

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
results = model.predict(test_file)

submit_file['count'] = results  
submit_file.to_csv(path + 'nolog_MaxAbs_relu.csv', index=False)  

'''                                 y값 로그O                               y값 로그X
결과정리                  일반레이어                      relu                 relu
                                                                            정확도고 자시고 -값때문에 
                                                                            다 리젝당함
안하고 한 결과 
loss값 :            1.4784865379333496          1.3939474821090698          20336.421875
R2 :                0.2917459249344957          0.3322434796939797          0.36312456133706306  
RMSE :              1.2159303183182582          1.1806555557714267          142.60581924576627

MinMax
loss값 :            1.4826191663742065          1.376928687095642           19986.966796875
R2 :                0.2897662439632225          0.3403961674900349          0.3740685146224785
RMSE :              1.217628490239822           1.1734260670034284          141.3752534752431

Standard
loss값 :            1.485615611076355           1.3592065572738647          20114.212890625  
R2 :                0.2883307752457015          0.3488858125246118          0.3700835382780253
RMSE :              1.2188583566891529          1.1658501305517504          141.82457038587344

Robust
loss값 :            1.47902512550354            1.3599188327789307          20291.291015625
R2 :                0.29148796673764255         0.34854455230668224         0.3645378718555361
RMSE :              1.2161517294252668          1.1661556116484861          142.44750081269214

MaxAbs
loss값 :            1.482558012008667           1.36235773563385            20105.78125
R2 :                0.2897954998115978          0.34737620891107013         0.3703474798313583
RMSE :              1.2176034117913292          1.167200855711969           141.7948542871914
'''