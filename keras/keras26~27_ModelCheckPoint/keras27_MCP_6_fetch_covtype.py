from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from pandas import get_dummies
import time

#1.데이터 로드 및 정제

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = get_dummies(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   #어떤 스케일러 사용할건지 정의부터 해준다.
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
#scaler.fit(x_train)       #어떤 비율로 변환할지 계산해줌.
#x_train = scaler.transform(x_train)   
#x_test = scaler.transform(x_test)    


#2. 모델구성,모델링

input1 = Input(shape=(54,))
dense1 = Dense(100,activation="relu")(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(60,activation="relu")(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(7,activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor="val_loss", patience=10, mode='min',verbose=1,baseline=None, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_6_fetch_covtype{krtime}_MCP.hdf5')

model.fit(x_train,y_train,epochs=10000, batch_size=100000,validation_split=0.11111111, callbacks=[es,mcp])


model.save(f"./_save/keras26_6_save_covtype{krtime}.h5")

#4. 평가 예측
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



print("======================= 2. load_model 출력 ======================")
model2 = load_model(f"./_save/keras26_6_save_covtype{krtime}.h5")
loss2 = model2.evaluate(x_test,y_test)
print('loss2 : ', loss2[0])
print('accuracy2 : ', loss2[1])



print("====================== 3. mcp 출력 ============================")
model3 = load_model(f'./_ModelCheckPoint/keras26_6_fetch_covtype{krtime}_MCP.hdf5')
loss3 = model3.evaluate(x_test,y_test)
print('loss3 : ', loss3[0])
print('accuracy3 : ', loss3[1])

'''
결과정리                일반레이어                  relu

안하고 한 결과                              epoch 1816
loss :              0.6296906471252441      0.3973493278026581          0.29210489988327026
accuracy :          0.7243468165397644      0.8348938226699829          0.8831881880760193      실수로 relu위치 바꿧는데 성능좋아짐.

MinMax                                     epoch 2559
loss :              0.6276273727416992      0.30850735306739807
accuracy :          0.7241231203079224      0.8762348890304565

Standard                                  에포 1400...
loss :              0.6274123787879944      0.2876836359500885
accuracy :          0.7250697016716003      0.8853740096092224

Robust                                    얼마나 좋은값 나오려고 1700에포를 함;
loss :              0.6272806525230408      0.28003185987472534
accuracy :          0.7248803973197937      0.888558030128479

MaxAbs
loss :              0.6282114386558533      0.2917717397212982
accuracy :          0.7231420874595642      0.8822071552276611
'''