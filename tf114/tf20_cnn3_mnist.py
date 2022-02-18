import tensorflow as tf, numpy as np,os
from keras.models import Sequential
from keras.layers import Conv2D
tf.compat.v1.set_random_seed(66)
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
w1 = tf.compat.v1.get_variable('w1', shape = [2, 2, 1, 16])    
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')    
L1 = tf.nn.relu(L1)   # activation 적용
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID

# model.add(Conv2D(filters=64, kernel_size = (2,2), strides=(1,1), 
#                   ,padding='valid',input_shape=(28,28,1)))
# print(w1)           # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
# print(L1)           # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)   x * w1에서 kernel에서 깍여서 27,27,64로 나온다.
# print(L1_maxpool)   # Tensor("MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

# Layers2 
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 16, 8])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')    
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID
# print(L2_maxpool)   # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)


# Layers3 
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 8, 4])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')    
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID
# print(L3_maxpool)   # Tensor("MaxPool_2:0", shape=(?, 4, 4, 64), dtype=float32)


# Flatten 
L_flat = tf.reshape(L3_maxpool,[-1,4*4*4])
# print(f"플래튼 : {L_flat}")     # (?, 128)

# Layer5    DNN
w5 = tf.compat.v1.Variable(tf.random.normal([64,32]))
b5 = tf.compat.v1.Variable(tf.random.normal([32]))

Hidden_layer1 =  tf.nn.relu(tf.matmul(L_flat,w5) + b5)

w6 = tf.compat.v1.Variable(tf.random.normal([32,10]))
b6 = tf.compat.v1.Variable(tf.random.normal([10]))

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer1,w6) + b6)


#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))   # categorical_crossentropy


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss)      
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000000005)   .minimize(loss)   


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 


while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,optimizer],feed_dict={x:x_train,y:y_train})
   
    print(f"{step:05d} \t{loss_val:.7f}")
    
    if loss_val < 0.2:       
        #4. 평가, 예측
        
        y_train_predict = sess.run(hypothesis, feed_dict={x:x_train})
        y_train_predict_int = sess.run(tf.math.argmax(y_train_predict,1))
        y_train_int = np.argmax(y_train,axis=1)
        train_acc = accuracy_score(y_train_int,y_train_predict_int)
        
        y_test_predict = sess.run(hypothesis, feed_dict={x:x_test})
        y_test_predict_int = sess.run(tf.math.argmax(y_test_predict,1))
        y_test_int = np.argmax(y_test,axis=1)
        test_acc = accuracy_score(y_test_int,y_test_predict_int)
        
        print(f"train_acc : {train_acc} test_acc : {test_acc}")
        # train_acc : 0.9405 test_acc : 0.9424 notbad... but too slow
        break
    
sess.close()