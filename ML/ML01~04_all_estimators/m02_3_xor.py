import numpy as np
from sklearn.svm import LinearSVC               # 선 긋는다. 선 그어서 분류한다.
from sklearn.linear_model import Perceptron     # 퍼셉트론을 가져왔다
from sklearn.metrics import accuracy_score

#1. 데이터  XOR
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
model = Perceptron()

#3. 훈련
model.fit(x_data,y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)
results = model.score(x_data,y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc)

# 우리눈엔 쉬워보이지만 이 간단한걸 못한다 
# xor 연산이란 같으면 0 = False, 다르면 1 = True를 반환해준다.