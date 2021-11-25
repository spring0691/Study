from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#훈련과 평가를 7:3으로 나누는데 임의로 나누어지게 완성지켜라
#1. 데이터 정제작업 
x = np.array(range(100))            0 1 2 3 4 5 6 7
y = np.array(range(1,101))          1 2 3 4 5 6 7

'''
x_train = random.sample(list(x), 30)
x_test = [x for x in x if x not in x_train]
y_train = list(x_train+int(1))
#y_test = 내가 작업하던곳... 근데 난 랜덤난수가지 생각못해서 할때마다 값이 바뀌었을거 같다.
랜덤난수 --> 하나의 Train-test set에서 여러번 훈련 돌려가면서 weight측정할때 오차 없게하기 위해 
랜덤난수 없이 반복훈련하면 다른 Train-test set 작업하는거랑 다를게없다 쉽게 말해서 train_test03 04 05 06 새로 하는 개념이다.
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, shuffle=True,
    random_state=66)

#print(x_test)   #[ 8 93  4  5 52 41  0 73 88 68]
#rint(y_test)
#랜덤 난수 넣어준다 -> 훈련을 반복해도 동일한 값이 나와야 제대로 된 훈련이 가능하기때문. 
#이게 없으면 한번 다시돌릴때마다 x_train~~y_test 값이 계속 바뀐다. 

#2. 모델링
model =  Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss) # 결과값에서 나온 loss?
result = model.predict([150])
print('[100]의 예측값 : ', result)

