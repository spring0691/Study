######### model = Sequential()  -> model = Model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np

#1.데이터 로드 및 정제

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링

input1 = Input(shape=(10,))
dense1 = Dense(100)(input1)
dense2 = Dense(80,activation="relu")(dense1)
dense3 = Dense(60)(dense2)
dense4 = Dense(40,activation="relu")(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1,outputs=output1)

# model = Sequential()
# model.add(Dense(100, input_dim=10))
# model.add(Dense(80,activation='relu')) #
# model.add(Dense(60))
# model.add(Dense(40,activation='relu')) #
# model.add(Dense(20))
# model.add(Dense(1))
# model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=10,validation_split=0.1111111, callbacks=[es])

model.save("./_save/keras25_2_save_diabets.h5")
#model = load_model("./_save/keras25_2_save_diabets.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)



'''
결과정리            일반레이어                      relu추가 

안하고 한 결과 
loss :             1808.4331                    1904.2423
r2   :             0.6439332274257212           0.6250691057093114

MinMax
loss :             1791.1332                    1938.3938
r2   :             0.6473394443221312           0.6183449003204229

Standard
loss :             1769.4728                    1996.4933
r2   :             0.6516041881978094           0.6069056263630164

Robust
loss :             1850.2620                    1953.5747
r2   :             0.6356974328243516           0.6153559346228505

MaxAbs
loss :             1843.4863                    2051.8716
r2   :             0.6370315636701395           0.5960019966708325
'''