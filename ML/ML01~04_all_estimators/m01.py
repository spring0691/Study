from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  

#1. 데이터 로드

datasets = load_iris()

x = datasets.data  
y = datasets.target
 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 모델구성

from sklearn.svm import LinearSVC   # 최초의 머신러닝을 쓸거다 제일 최초모델

model = LinearSVC()     # 모델구성 끝. 끝? 뭐 더 안해?  퍼셉트론이 1개다 

#3. 컴파일, 훈련

model.fit(x_train,y_train)      # 모델구성에 훈련까지 다 포함되어 있어서 fit하면 끝이다..

#4. 평가, 예측

result = model.score(x_test,y_test)   # 알아서 자기가 분류모델인지 회귀모델인지 판단하고 acc아니면 r2로 값을 준다.
# 머신러닝 모델들도 딱 2종류밖에 없다. 분류모델과 회귀모델

from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)

print("result : ", result)
print('accuracy_score : ', acc)