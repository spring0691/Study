from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,MaxPool2D,Conv2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


#1. 데이터 로드 및 전처리

# np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])     
# np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])     
# np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])      
# np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1]) 

x_train = np.load('./_save_npy/keras47_5_train_x.npy')      #(160, 150, 150, 3)
y_train = np.load('./_save_npy/keras47_5_train_y.npy')      #(160,)
x_test = np.load('./_save_npy/keras47_5_test_x.npy')        #(120, 150, 150, 3)
y_test = np.load('./_save_npy/keras47_5_test_y.npy')        #(120,)

#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#print(x_train)
#print(x_train.shape)   금방금방 로드되는것을 확인할수있다.

#2. 모델링

model = Sequential()
model.add(Conv2D(32, (2,2),input_shape=(150,150,3)))                                           
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일,훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor ="val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)
model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_split=0.2, callbacks=[es])  

#4. 평가,예측

loss = model.evaluate(x_test, y_test, batch_size=1)

print("----------------------loss & accuracy-------------------------")
print(' loss : ',round(loss[0],4))
print(' acc : ', round(loss[1],4))