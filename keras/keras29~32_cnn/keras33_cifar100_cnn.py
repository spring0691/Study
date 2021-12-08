from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import cifar100 # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

#1.데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler =StandardScaler()   #MinMaxScaler()RobustScaler()MaxAbsScaler()

x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

#2.모델링
model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(100, activation='softmax'))

#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


es = EarlyStopping(monitor="val_accuracy", patience=50, mode='max',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras32_cifar10_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1000,validation_split=0.2, callbacks=[es])#,mcp

#model.save(f"./_save/keras32_save_cifar10.h5")

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#            기본                       기본+Minmax       기본 + standard
# loss :     5.560264587402344          3.04172852      3.0337114334106445
# accuracy : 0.00930000003427267        0.2642          0.2630000114440918


