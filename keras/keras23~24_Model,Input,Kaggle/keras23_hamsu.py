import numpy as np
#1. 데이터
x = np.array([range(100), range(301,401), range(1,101)])        # 100,3 인데 현재 3,100
y = np.array([range(71,81)])
print(x.shape, y.shape)     # (3,100) (2,100)
x = np.transpose(x)
y = np.transpose(y)
# print(x.shape, y.shape)     # (100,3) (100,2)
# x = x.reshape(1,10,10,3)
# print(x.shape, y.shape)     # (1, 10, 10, 3) (10, 1)

#2. 모델구성        행의 개수는 같아야 하지만 열의 개수는 달라도 된다. 열 = 특성,피쳐,등등
from tensorflow.keras.models import Sequential, Model # 함수형모델 Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(3,))      # input 칼럼 입력해주고
dense1 = Dense(10)(input1)      # input1에서 받아왔다
dense2 = Dense(9)(dense1)       # dense1에서 받아왔다 이런식으로 연결.
dense3 = Dense(8, activation='relu')(dense2)    # 이런식으로 activaiton적용       
output1 = Dense(1)(dense3)      
model = Model(inputs=input1,outputs=output1)   #함수형 모델 inputs시작과 outputs 끝을 지정해준다.
model.summary()

"""
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3)]               0            input 정보도 표시해준다
_________________________________________________________________
dense (Dense)                (None, 10)                40
dense_1 (Dense)              (None, 9)                 99
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 80
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
=================================================================
"""
# model = Sequential()                    # 모델을 정의하고 add로 층을 쌓는다.
# model.add(Dense(10, input_dim=3))       # x는 (100,3)이다   행 무시 (N,3)
# #model.add(Dense(9, input_shape=(3,)))    # 차원이 늘어나면 행(x값)다 때고 열 columns 값만 쓴다.
# model.add(Dense(8))
# model.add(Dense(2))
# model.summary()
'''
dense (Dense)                (None, 10)                40
x자리가 None인 이유 = 행의 개수는 신경쓰지 않겠다. 상관없다.
'''

# 이미지같은 경우는 3차원 모델이라서 다른 방식으로 모델링 해줘야한다. 가로,세로,그리고 컬러(rgb겹침, tensor) 그리고 여러장이니까 + 1차원해서 총 4차원

