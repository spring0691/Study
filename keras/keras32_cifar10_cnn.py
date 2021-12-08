from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import cifar10 # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)
#plt.imshow(x_train[2],'gray')  자료 확인
#plt.show()

#print(np.unique(y_train, return_counts=True))   # 0~9까지 각각 5000개씩 10개의 label값.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(y_train.shape,y_test.shape)


#scaler = MinMaxScaler()   #어떤 스케일러 사용할건지 정의부터 해준다.
scaler = StandardScaler()
x_train= x_train.reshape(50000,-1)  # 4차원 (50000,32,32,3)을 가로로 1자로 쫙펴준다.  행 세로 열 가로   (50000,3072)
x_test = x_test.reshape(10000,-1)

scaler.fit(x_train) # 비율을 가져옴

x_train = scaler.transform(x_train)  # 스케일러 비율이 적용되서 0~1.0 사이로 값이 다 바뀜 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)


# x_train_trans = scaler.transform(x_train_reshape) 
# x_train = x_train.reshape(x_train.shape)

# x_test_reshape = x_test.reshape(10000,-1)
# x_test_trans = scaler.transform(x_test_reshape)
# x_test = x_test_trans.reshape(x_test.shape)

#2. 모델링

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
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras32_cifar10_1_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1000,validation_split=0.2, callbacks=[es,mcp])#

model.save(f"./_save/keras32_save_cifar10_1.h5")
#model = load_model("./_save/keras32_save_cifar10.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#            기본                   기본+Minmax
# loss :     1.0126644372940063     1.1994456052780151
# accuracy : 0.6455000042915344     0.5812000036239624

