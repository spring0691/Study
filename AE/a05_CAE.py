# 2번 카피해서 복붙, CNN으로 딥하게 구성
# Conv2D, MaxPool, Conv2D, MaxPool, Conv2D                                  -> Encoder
# Conv2D, UpSampling2D, Conv2D, UpSampling2D, Conv2D, UpSampling2D, Conv2D  -> Decoder

import numpy as np
from sympy import Max
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

# autoencoder를 함수로 제작하여 사용해보기.

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size,kernel_size=(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size*2,(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size*4,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size*2,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(1,(2,2),padding='same',activation='sigmoid'))
    return model

def show_img(original,decoded_img,n):
    import matplotlib.pyplot as plt
    
    n = n
    plt.figure(figsize=(20,4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(original[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_img[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.show()

# def autoencoder(hidden_layer_size):
#     model = Sequential([
#         Dense( units=hidden_layer_size, input_shape= ( x_train.shape(1), ), activation='relu'),
#         Dense( units=x_train.shape(1), activation='sigmoid') 
#     ])
#     return model

#1. 데이터
(x_train, _) , (x_test, _)  = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float')/255   # 또는 /255. 으로 나눠도 된다.
x_test = x_test.reshape(10000,28,28,1).astype('float')/255   

#2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

model = autoencoder(hidden_layer_size=32)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train,x_train,epochs=10)

#4. 평가, 예측
output = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2,5,figsize = (20,7))      # \는 줄바꿈 위해

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()