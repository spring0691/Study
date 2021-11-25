from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt     # 그래프나 그림그릴때 많이 씀
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,17, 8,14,21, 9, 6,19,23,21])

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train,y_train,epochs=100, batch_size=1)

#4. 평가, 예측      이건 진짜 말그대로 그냥 해보는것. 여기서 나온 loss값과 훈련에서 나온 loss값의 차이가 적을수록 정확한것.
# 근데 생각해보니까 기본적으로 데이터를 정제 잘하고 뺄건 빼서 넘겨주는게 제일 중요할것 같다.
loss = model.evaluate(x_test,y_test) # 평가해보는 단계. 이미 다 나와있는  w,b에 test데이터를 넣어보고 평가해본다.
print('loss : ', loss)

y_predict = model.predict(x)

plt.scatter(x, y) # scatter 흩뿌리다 그림처럼 보여주다?
plt.plot(x, y_predict, color='red') # scatter 점찍다 plot 선을 보여준다 
plt.show() 