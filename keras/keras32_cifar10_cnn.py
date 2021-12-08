from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import cifar10 # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

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



#2. 모델링

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras32_cifar10_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1000,validation_split=0.2, callbacks=[es])#,mcp

model.save(f"./_save/keras32_save_cifar10.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :     
# accuracy : 

