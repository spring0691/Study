# 지금까지 배운 기존의 Train & Test 시스템은 훈련 후 바로 실전을 치뤄서 실전에서 값이 많이 튀었다.
# 이걸 더 보완하기위해서 Train 후 validation(검증) 거치는 걸 1 epoch로 하여 계속반복하며 값을 수정 후
# 실전 Test를 치루는 방식으로 더 개선시킨다. 앞으로는 데이터를 받으면 train validation test 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split # 싸이킷런, 싸이킷런 

#1. 데이터 정제작업
x = np.array(range(1,17))
y = np.array(range(1,17))

# train_test_split로 나누시오 10 3 3 
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8125, random_state=66)
# train&test를 13개 3개로 나누어 준다  
#x_val,x_test,y_val,y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=49)

# print(x_train)
# print(y_train)
# print(x_val,y_val)
# print(x_test)
# print(y_test)
 #확인작업


# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          #validation_data=(x_val, y_val))
           validation_split=0.3)

# 여기 이 validation_split=0.3 기능 하나로 그 전과 비슷하게 x_train을 다시 train과 
# validaion으로 알아서 비율까지 쪼개준다. 
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)
