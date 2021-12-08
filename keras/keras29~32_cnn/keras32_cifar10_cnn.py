from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import cifar10 # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from pandas import get_dummies
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import ssl      
ssl._create_default_https_context = ssl._create_unverified_context      # 인터넷 연결오류 해결.
#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
#print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)

#plt.imshow(x_train[2],'gray')  자료 확인
#plt.show()

#print(np.unique(y_train, return_counts=True))   # 0~9까지 각각 5000개씩 10개의 label값.

# 3가지 방법으로 onehotencoding 해봄.
enco = OneHotEncoder(sparse=False)
y_train = enco.fit_transform(y_train.reshape(-1,1))    # -1,1로 reshape한다는건 세로로 쭉 나열하겠다는 뜻. 행렬변환
#y_train = y_train.reshape(len(y_train),)      #get_dummies는 1차원데이터만 사용가능하다. Data must be 1-dimensional
#y_train = get_dummies(y_train)         #그래서 reshape로 50000행1열을 쪼개서 50000개의 스칼라값으로 만들어줬다.
y_test = to_categorical(y_test)

#print(y_train[2],y_test[2])
#print(y_train.shape,y_test.shape)

#4차원 데이터의 스케일러 적용하는 방법

scaler =   StandardScaler()#MinMaxScaler()RobustScaler()MaxAbsScaler()    어떤 스케일러 사용할건지 정의부터 해준다.

#x_train= x_train.reshape(50000,-1)  # 4차원 (50000,32,32,3)을 가로로 1자로 쫙펴준다.  행 세로 열 가로   (50000,3072)
#x_test = x_test.reshape(10000,-1)

#scaler.fit(x_train) 비율을 가져옴

#x_train = scaler.transform(x_train) 스케일러 비율이 적용되서 0~1.0 사이로 값이 다 바뀜 
#x_test = scaler.transform(x_test) 

#x_train = x_train.reshape(50000, 32,32,3) 그리고 다시 원래의 배열대로 되돌려준다. 
#x_test = x_test.reshape(10000, 32,32,3)

# 위의 일련의 작업들을 2줄로 압축하면 이렇게 줄일수 있다.
x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

#2. 모델링

# model = Sequential()
# model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(32,32,3), activation='relu'))
# # strides = 
# model.add(MaxPooling2D())
# model.add(Conv2D(10,(2,2), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.5))
# model.add(Conv2D(10,(2,2), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Flatten())       
# model.add(Dense(64))
# model.add(Dropout(0.5))
# model.add(Dense(16))
# model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 


#es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras32_cifar10_stardard_MCP.hdf5')
#model.fit(x_train,y_train,epochs=10000, batch_size=1000,validation_split=0.2, callbacks=[es,mcp])#

#model.save(f"./_save/keras32_save_cifar10_standard.h5")
model = load_model("")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#            기본                   기본+Minmax             기본+satndard 
# loss :     1.0126644372940063     1.1994456052780151      0.3034297525882721      1.202593207359314
# accuracy : 0.6455000042915344     0.5812000036239624      0.8963000178337097      0.5715000033378601

