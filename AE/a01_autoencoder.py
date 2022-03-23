# 데이터에 특정 처리를 가한 후에 그 이미지를 다시 복호화 시켜서 원본 -> 새로운 원본의 형태로 재 생성한다. 비지도학습 개념.
# 앞뒤가 똑같다 GAN의 이전 개념.

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

#1. 데이터
(x_train, _) , (x_test, _)  = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255   # 또는 /255. 으로 나눠도 된다.
x_test = x_test.reshape(10000, 784).astype('float')/255   

#2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))

# encoded = Dense(64, activation='relu')(input_img)       # 중간 encoded의 값에 따라 output의 흐림한 정도가 결정된다.
# encoded = Dense(16, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)
# encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(154, activation='relu')(input_img)        # 전에 PCA했을때 원본 정확도 0.95이상 유지되는 지점
# encoded = Dense(486, activation='relu')(input_img)          # PCA 원본 0.999이상 유지되는 지점

decoded = Dense(784, activation='sigmoid')(encoded)     # linear도 가능 근데 통상적으로 sigmoid + binary_crossentropy 사용함

# 원본을 작게했다가 크게 하는과정에서 중요한 feature만 남기고 나머지 잡티,그을림,깨짐같은 불필요한 정보가 삭제되었다가
# 원상복구하는 과정에서 중요 feature만 그대로 들어가고 불필요 noise는 날아간채로 복구되어 좋은 효과를 얻게 된다.
# 살짝 흐릿하게 output이 나온다. (줄였다가 원복하는 과정에서 나오는 부작용.)
# 큰 픽셀의 이미지에서는 효과가 좋지만 작은 픽셀의 이미지 같은 경우는 효과를 장담할 수 없다.

autoencoder = Model(input_img, decoded)

#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')                 차이 비교해보기

autoencoder.fit(x_train,x_train, epochs=30, batch_size=128, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

# 창작물은 눈으로 확인해야함. loss가 무의미하다.

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()