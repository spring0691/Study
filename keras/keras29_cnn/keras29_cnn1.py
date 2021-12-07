from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, Activation


model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2), input_shape=(10,10,1)))   #<-- img를 받기위해 사용. 10은 그 다음레이어로 전달할 값
            # kernel_size=(2,2)  사진을 2,2로 쪼개서 작업하겠다. # Conv2D 할때는 5,5,1하더라도 1을 입력해야한다. 차원 구분하기위해
model.add(Conv2D(5,(3,3), activation='relu'))
model.add(Conv2D(7,(2,2), activation='relu'))
model.add(Flatten())        #<-- 위에서 넘겨주는 값을 일렬로 쭉 나열해서 1개의 값으로 만들어준다.
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #      10,kernel_size=(2,2), input_shape=(5,5,1) 
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50           <-- 왜 50개?
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 3, 3, 5)           205
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 2, 2, 7)           147
=================================================================
Total params: 402
Trainable params: 402
Non-trainable params: 0
_________________________________________________________________
'''