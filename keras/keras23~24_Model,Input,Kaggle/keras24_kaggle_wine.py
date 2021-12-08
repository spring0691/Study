from tensorflow.keras.models import Sequential, Model         
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
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
#labelencoder로 변환해줘야함을 알수있다. 또한 quality는 결과치이므로 y값으로 분리시켜야한다.


x = train.drop(['id','quality'], axis=1)    # id와 quality열 제거

Le = LabelEncoder()     # 함수 선언
#라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지의 연속적 수치 데이터로 표현
label = x['type']   # label안에 x의 type열 값들 저장.   x.type과 x['type']은 같다. 
Le.fit(label)       # fit으로 범주를 찾아낸다.
x['type'] = Le.transform(label)     # 라벨인코더로 type값 변환
#print(type(x.type))     # pandas.core.series.Series
#print(x.type)          # 값이 바뀐것을 확인.
#print(x.type.info())    # pandas.core.series.Series 는 값이 찍히지 않는다. 그냥 이 자체로 이해하는게 편할거같다.
#print(x.type[2:3] + x.type[3:4])   덧셈 연산 안된다. 숫자가 아니라 문자로 바뀌는거 같다.
#print(x.type.value_counts())    #x의 type열의 개수를 세주는 기능 value_counts()
# categorical 형 데이터의 value별로 개수를 카운트해준다.
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

#print(y[:50])            # 앞에서부터 4,5,6,7,8순으로 잘 변환되어있다.


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=66)  


#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
    
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    
test_file = scaler.transform(test_file)





#2. 모델링

input1 = Input(shape=(12,))
dense1 = Dense(60, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(40, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(20, activation='relu')(drop2)
dense4 = Dense(10, activation='relu')(dense3)
output1 = Dense(5, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
      
# model = Sequential()
# model.add(Dense(100, input_dim=12))    
# model.add(Dense(80, activation='relu')) # 
# model.add(Dense(60)) #
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu')) 
# model.add(Dense(20))
# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(5))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=5000,batch_size=5, verbose=1,validation_split=0.1111111111,callbacks=[es])

#4. 평가
loss = model.evaluate(x_test,y_test)   
print('loss값 accuracy값 : ', loss) 



############################# 제출용 제작 ####################################
results = model.predict(test_file)

results_int = np.argmax(results, axis=1).reshape(-1,1) + 4 # 0부터되돌려주므로 4더해준다.
# argmax 중요 ! argmax란 그니까 원핫인코딩된 데이터를 결과데이터에 넣을때 다시 숫자로,
# 되돌려 주는 편리한 기능을 제공해주는 함수이다. colums값이 꽃의 종류같이 문자일경우
# 내가 작업하거나 다른 함수기능으로 한번더 디코딩 해줘야할거 같다.
# [0.1, 0.4, 0.3, 0.2] -> 0.4가 제일 크다 -> [0,1,0,0] -> 2로 반환 완전 편하당 
# padas는 value_count 기능으로 numpy는 np.unique으로 안의 값들을 정리된 상태로 편하게 이해 할 수 있다.
#print(np.unique(results_int))  results_int안에 담긴 값들의 unique값을 확인 할 수 있다.

submit_file['quality'] = results_int

acc= str(round(loss[1], 4)).replace(".", "_")
submit_file.to_csv(path+f"result/accuracy_{acc}.csv", index = False)


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
