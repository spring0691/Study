import tensorflow as tf, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
tf.compat.v1.set_random_seed(66)

#1. 데이터 
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

from keras.utils import to_categorical  # OnehotEncoder은 0부터 인코딩함
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32, [None,28,28,1])
y = tf.compat.v1.placeholder(tf.float32, [None,10])

#2. 모델구성

w1 = tf.compat.v1.get_variable('w1', shape = [2, 2, 1, 64])    # 2,2는 kernel_size고 1은 채널값, 64는 output값
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')    # strides에서 [1,a,b,1]에서 1은 자리 채워주는 용도이고, a,b는 입력값이다.

# model.add(Conv2D(filters=64, kernel_size = (2,2), strides=(1,1), 
#                   ,padding='valid',input_shape=(28,28,1)))

# print(w1)     # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
# print(L1)     # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)   x * w1에서 kernel에서 깍여서 27,27,64로 나온다.

