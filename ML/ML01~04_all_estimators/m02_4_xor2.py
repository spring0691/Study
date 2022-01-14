import numpy as np
from sklearn.svm import LinearSVC,SVC               # SVC 넌 도대체 뭐냐? 어떻게 잡은거지? 다항식, 곡선?
from sklearn.linear_model import Perceptron         
from sklearn.metrics import accuracy_score

#1. 데이터  XOR
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()
model = SVC()                                       # 넌 뭐니 SVC야?

#3. 훈련
model.fit(x_data,y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)
results = model.score(x_data,y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc)

# 이걸 잡네