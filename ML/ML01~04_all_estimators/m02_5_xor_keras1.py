import numpy as np
from sklearn.svm import LinearSVC,SVC               
from sklearn.linear_model import Perceptron         
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터  XOR
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()
# model = SVC()                                     
model = Sequential()                                
model.add(Dense(1, input_dim=2, activation='sigmoid'))  # input이 2개 output이 1개 이게 sigmoid로 activation 받아서 output이 나온다.

#3. 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data,y_data,batch_size=1, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)

results = model.evaluate(x_data,y_data)

print(x_data, "의 예측결과 : ", y_predict)
print("metrics_acc : ", results[1])

acc = accuracy_score(y_data, np.round(y_predict,0).astype(int))
print("accuracy_score : ", acc)

# Sequential 써서 Dense를 썼는데도 단층퍼셉트론으로는 Xor을 못잡는다