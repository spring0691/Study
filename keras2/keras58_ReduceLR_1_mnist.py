import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.

model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3), input_shape=(28,28,1)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
optimizer = Adam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['acc'])

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=100,validation_split=0.2, callbacks=[es])

result = model.evaluate(x_test,y_test,batch_size=32)

print(result)