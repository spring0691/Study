# 파라미터의 수
# (3, 3) 필터 한개에는 3 x 3 = 9개의 파라미터가 있음(Numpy연산 방식으로 이해)
# 그리고 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 3이 곱해짐
# 그리고 Conv2D(32, ...) 라면 32는 32개의 필터를 적용하여 다음 층에서는 채널이 총 32개가 되도록 만든다는 뜻
# 여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 32개가 추가로 더해짐

# ex) 3 x 3(필터 크기) x 3 (입력 채널(RGB, 흑백이면 1)) x 32(#출력 채널) + 32(출력 채널 bias) = 896
# model.add(Conv2D(a, kernel_size=(b,c), input_shape=(d, e, f)))
# a = filters or kernel
# Filter와 Kernel은 같음 ex) (b,c) -> kernel_size -> 파라미터 연산할때는 b*c값을 사용함 필터크기.
# d, e, 
# f = channel : 컬러 이미지는 3개의 채널로 구성됨. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됨 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, Activation, MaxPooling2D


model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1 ,padding='same',input_shape=(10,10,1)))   #<-- img를 받기위해 사용. 10은 그 다음레이어로 전달할 값 출력값.
                                # kernel_size=(2,2)  사진을 2,2로 쪼개서 작업하겠다. # Conv2D 할때는 5,5,1하더라도 1을 입력해야한다. RGB구분 위해.
                                # padding='same'는 겉에 0값으로 둘러싸서 kerner_size로 쪼개도 row,col값을 유지시켜준다. default는 valid. -> 유지시켜주지 않음
model.add(MaxPooling2D())       # dropout과 비슷한개념 conv2d가 knrnel을 이용해서 중첩시키며 특성을 추출해나간다면 maxpoolig은 픽셀을 묶어서 그중에 가장큰값만 뺀다.
                                # maxpooling는 값을 반으로 계속 줄여나간다. default 2,2=4픽셀당 1개값. 
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
conv2d (Conv2D)              (None, 9, 9, 10)          50           (2*2(kennel_size)*1(channel)+1(bias)) * 10(filters)
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 5)           455          (3*3(kennel_size)*10(channel)+1(bias)) * 5(filters)
                                                                    여기서 채널은 전 레이어의 input_shape의 채널값을 받아옴.
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 7)           147          (2*2*5+1) * 7 
_________________________________________________________________
flatten (Flatten)            (None, 252)               0
_________________________________________________________________
dense (Dense)                (None, 64)                16192        252 * 64 + 64 = 16192
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

첫번째 인자 : 컨볼루션 필터의 수 입니다.
두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.
padding : 경계 처리 방법을 정의합니다.
‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
(행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
activation : 활성화 함수 설정합니다.
‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

'''