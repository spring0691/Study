from tensorflow.keras.models import Sequential, Model         
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies

#1. 데이터 로드 및 정제
path = "../_data/dacon/wine/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sample_Submission.csv')     

#일단 기본적으로 csv파일 직접보거나 excel로 열어서 colums값을 직접 보고 분석하는게 좋다.
#print(train.info())
#print(train)   값 들을 보면 id는 그냥 단순 번호라서 삭제해야하고 type는 white와 red로 되어있어서 
#               labelencoder로 변환해줘야함을 알수있다. 또한 quality는 결과치이므로 y값으로 분리시켜야한다.


x = train.drop(['id','quality'], axis=1)    # id와 quality열 제거

Le = LabelEncoder()
#라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지의 연속적 수치 데이터로 표현
label = x['type']
Le.fit(label)
x['type'] = Le.transform(label)     # 라벨인코더로 type값 변환

#print(x)   #확인        여기까지가 x값 정제.
#print(x.shape)       (3231, 12)   


test_file = test_file.drop(['id'], axis=1)
label2 = test_file['type']
Le.fit(label2)
test_file['type'] =Le.transform(label2)
#print(test_file)   #id값 사라진거 확인. type 1과0으로 바뀐거확인

y = train['quality']    # train에서 quality 열 값만 가져오겠다.
#print(y.unique())       # 6,7,5,8,4     5가지 값만 나온다. 다중분류모델 고고~
y = get_dummies(y)
y = np.log1p(y) 

#print(y[:50])            # 앞에서부터 4,5,6,7,8순으로 잘 변환되어있다.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)  

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
scaler = RobustScaler()
#scaler = MaxAbsScaler()
    
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    
test_file = scaler.transform(test_file)


#2. 모델링

# input1 = Input(shape=(8,))
# dense1 = Dense(16)(input1) #,activation="relu"
# dense2 = Dense(24)(dense1)
# dense3 = Dense(32)(dense2) #,activation="relu" 
# dense4 = Dense(16)(dense3)
# dense5 = Dense(8)(dense4)
# output1 = Dense(1)(dense5)
# model = Model(inputs=input1,outputs=output1)
      
model = Sequential()
model.add(Dense(60, activation='relu', input_dim=12))    
model.add(Dense(48, activation='relu')) #, activation='relu'
model.add(Dense(36)) 
model.add(Dense(24)) #, activation='relu'
model.add(Dense(12))
model.add(Dense(5,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=5000,batch_size=50, verbose=1,validation_split=0.11111111,callbacks=[es])

#4. 평가
loss = model.evaluate(x_test,y_test)   
print('loss값 accuracy값 : ', loss) 





############################# 제출용 제작 ####################################
results = model.predict(test_file)

results_int = np.argmax(results, axis=1).reshape(-1,1) + 4 # 0부터되돌려주므로 4더해준다.
# argmax 중요 !

submit_file['quality'] = results_int

submit_file.to_csv(path + 'Roubst_relu3.csv', index=False)  


'''                                                                                   
결과정리                  일반레이어                      relu                   
                                                                                    
안하고 한 결과 
loss값 :                                            1.0440469980239868
accuracy :                                          0.5123456716537476

MinMax
loss값 :                                            0.9981087446212769
accuracy :                                          0.5432098507881165

Standard
loss값 :                                            0.9914785027503967
accuracy :                                          0.540123462677002

Robust
loss값 :                                            1.031555414199829       1.0050442218780518
accuracy :                                          0.5617284178733826      0.5771604776382446

MaxAbs
loss값 :                                            1.0235952138900757
accuracy :                                          0.5246913433074951
''' 
