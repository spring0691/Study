######### model = Sequential()  -> model = Model

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 미리 처리한다 -> 전처리
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
import time

#1.데이터 로드 및 정제

datasets = load_boston()
x = datasets.data
y = datasets.target


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)    # train과 test로 나누고나서 스케일링한다.

##################################### 스케일러 설정 옵션 ########################################
#scaler = MinMaxScaler()   
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)       
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링

input1 = Input(shape=(13,))
dense1 = Dense(50)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(15,activation="relu")(dense2)
dense4 = Dense(8,activation="relu")(dense3)
dense5 = Dense(5)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1,outputs=output1)

# model = Sequential()
# model.add(Dense(50, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(15,activation="relu")) #
# model.add(Dense(8,activation="relu")) #
# model.add(Dense(5))
# model.add(Dense(1))
#model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping  
es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=10,validation_split=0.111111, callbacks=[es])

#time = time.time()
#tm = time.gmtime(time)
#print(time)
model.save("./_save/keras25_1_save_boston.h5")
#model = load_model("./_save/keras25_1_save_boston.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)


'''
결과정리            일반레이어                  relu추가                                          

안하고 한 결과                             
loss :            31.4710                   27.8730
r2   :            0.6822831693844669        0.7186065480994359

MinMax
loss :            28.5702                   23.6224
r2   :            0.7115684464061318        0.7615192701093052

Standard
loss :            31.4710                   13.9806
r2   :            0.6822826358440421        0.8588578155492782

Robust
loss :            31.3751                   16.9108
r2   :            0.6832512364127951        0.8292758037688225

MaxAbs
loss :            30.5554                   23.8010
r2   :            0.6915265579923042        0.7597155128379716
'''