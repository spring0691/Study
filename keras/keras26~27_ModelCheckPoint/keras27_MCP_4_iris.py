from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import numpy as np
from pandas import get_dummies
import time
#1.데이터 로드 및 정제

datasets = load_iris()
x = datasets.data
y = datasets.target

y = get_dummies(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)      
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    

#2. 모델구성,모델링

input1 = Input(shape=(4,))
dense1 = Dense(70)(input1)
dense2 = Dense(55)(dense1)
dense3 = Dense(40,activation="relu")(dense2)
dense4 = Dense(25)(dense3)
dense5 = Dense(10,activation="relu")(dense4)
output1 = Dense(3,activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)



#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_4_iris{krtime}_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1,validation_split=0.11111111, callbacks=[es,mcp])

model.save(f"./_save/keras26_4_save_iris{krtime}.h5")


#4. 평가 예측
print("======================= 1. 기본 출력 =========================")

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

print("======================= 2. load_model 출력 ======================")
model2 = load_model(f"./_save/keras26_4_save_iris{krtime}.h5")
loss2 = model2.evaluate(x_test,y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)

r2 = r2_score(y_test,y_predict2) 
print('r2스코어 : ', r2)

print("====================== 3. mcp 출력 ============================")
model3 = load_model(f'./_ModelCheckPoint/keras26_4_iris{krtime}_MCP.hdf5')
loss3 = model3.evaluate(x_test,y_test)
print('loss3 : ', loss3)

y_predict3 = model3.predict(x_test)

r2 = r2_score(y_test,y_predict3) 
print('r2스코어 : ', r2)

'''
결과정리            일반레이어                      relu

안하고 한 결과 
loss :          0.0033079693093895912       0.0010957516497001052
accuracy :      1.0                         1.0

MinMax
loss :          0.007345080375671387        0.0005389913567341864
accuracy :      1.0                         1.0

Standard
loss :          0.0014144571032375097       0.011362032033503056
accuracy :      1.0                         1.0

Robust
loss :          0.0014624781906604767       0.0034313490614295006
accuracy :      1.0                         1.0

MaxAbs
loss :          0.0035078583750873804       0.0006933937547728419
accuracy :      1.0                         1.0
'''