# 파라미터의 수
# (3, 3) 필터 한개에는 3 x 3 = 9개의 파라미터가 있음(Numpy연산 방식으로 이해)
# 그리고 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 3이 곱해짐
# 그리고 Conv2D(32, ...) 라면 32는 32개의 필터를 적용하여 다음 층에서는 채널이 총 32개가 되도록 만든다는 뜻
# 여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 32개가 추가로 더해짐
# ex) 3 x 3(필터 크기) x 3 (입력 채널(RGB, 흑백이면 1)) x 32(#출력 채널) + 32(출력 채널 bias) = 896

# model.add(Conv2D(a, kernel_size=(b,c), input_shape=(d, e, f)))
# a = 출력채널
# Filter와 Kernel은 같음 ex) (b,c) -> kernel_size
# d, e, 
# f = channel : 컬러 이미지는 3개의 채널로 구성됨. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됨 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, Activation


model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2), input_shape=(10,10,1)))   #<-- img를 받기위해 사용. 10은 그 다음레이어로 전달할 값 출력값.
            # kernel_size=(2,2)  사진을 2,2로 쪼개서 작업하겠다. # Conv2D 할때는 5,5,1하더라도 1을 입력해야한다. RGB구분 위해.
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
conv2d (Conv2D)              (None, 9, 9, 10)          50           커널사이즈=패널   4x1(입력데이터의채널값)x10(출력채널)+10 = 50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455          9(3,3패널의 파라미터값)x10(입력채널-전레이어의출력채널)x5
                                                                    (출력채널)+5(출력채널만큼의바이어스값)  = 455
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147          4 x 5 x 7 + 7 = 147
_________________________________________________________________
flatten (Flatten)            (None, 252)               0
_________________________________________________________________
dense (Dense)                (None, 64)                16192        252 * 64 + 4 = 16192
_________________________________________________________________
dropout (Dropout)            (None, 64)                0            
_________________________________________________________________
dense_1 (Dense)              (None, 16)                1040
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 85
=================================================================
Total params: 17,969
Trainable params: 17,969
Non-trainable params: 0
_________________________________________________________________
'''

# Output Size = (W - F + 2P) / S + 1
# W: input_volume_size
# F: kernel_size
# P: padding_size
# S: strides