from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])      # 훈련
x_test = np.array([8,9,10])             # 평가
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss) # 결과값에서 나온 loss?
result = model.predict([11])
print('[11]의 예측값 : ', result)