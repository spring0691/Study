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

# Layer1 
w1 = tf.compat.v1.get_variable('w1', shape = [2, 2, 1, 128])    
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')    

L1 = tf.nn.relu(L1)   # activation 적용
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID

# model.add(Conv2D(filters=64, kernel_size = (2,2), strides=(1,1), 
#                   ,padding='valid',input_shape=(28,28,1)))

print(w1)           # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)           # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)   x * w1에서 kernel에서 깍여서 27,27,64로 나온다.
print(L1_maxpool)   # Tensor("MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

# Layers2 
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')    
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID
print(L2_maxpool)   # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# Layers3 
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')    
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID
print(L3_maxpool)   # Tensor("MaxPool_2:0", shape=(?, 4, 4, 64), dtype=float32)

# Layers4 
w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())    # 각 커널에 들어가는 값에 대한 초기화
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='SAME')    
L4 = tf.nn.elu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID
print(L4_maxpool)   #Tensor("MaxPool_3:0", shape=(?, 2, 2, 32), dtype=float32)

# Flatten 
L_flat = tf.reshape(L4_maxpool,[-1,2*2*32])