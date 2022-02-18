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
w1 = tf.compat.v1.get_variable('w1', shape = [2, 2, 1, 128])    
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')    
L1 = tf.nn.relu(L1)   # activation 적용
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID

# model.add(Conv2D(filters=64, kernel_size = (2,2), strides=(1,1), 
#                   ,padding='valid',input_shape=(28,28,1)))
# print(w1)           # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
# print(L1)           # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)   x * w1에서 kernel에서 깍여서 27,27,64로 나온다.
# print(L1_maxpool)   # Tensor("MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

# Layers2 
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')    
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID
# print(L2_maxpool)   # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)


# Layers3 
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')    
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 2,2는 maxpool값.   max_pool default = VALID
# print(L3_maxpool)   # Tensor("MaxPool_2:0", shape=(?, 4, 4, 32), dtype=float32)

# Layer4
w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='SAME')  
L4 = tf.nn.elu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# print(L4_maxpool)   # Tensor("MaxPool_3:0", shape=(?, 2, 2, 32), dtype=float32)


# Flatten 
L_flat = tf.reshape(L3_maxpool,[-1,2*2*32])
# print(f"플래튼 : {L_flat}")     # (?, 128)

# Layer5    DNN
w5 = tf.compat.v1.get_variable('w5',shape=[2*2*32, 64], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.compat.v1.Variable(tf.random.normal([64]), name='b5')
L5 =  tf.nn.selu(tf.matmul(L_flat,w5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=0.5)   # `rate = 1 - keep_prob`

# Layer6    DNN
w6 = tf.compat.v1.get_variable('w6',shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.compat.v1.Variable(tf.random.normal([32]), name='b6')
L6 =  tf.nn.relu(tf.matmul(L5,w6) + b6)
L6 = tf.nn.dropout(L6, rate=0.5)

# Layer7    DNN
w7 = tf.compat.v1.get_variable('w7',shape=[32, 10], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.compat.v1.Variable(tf.random.normal([10]), name='b6')
L7 =  tf.nn.relu(tf.matmul(L6,w7) + b7)
L7 = tf.nn.dropout(L7, rate=0.5)
hypothesis = tf.nn.softmax(L7)


#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))   # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)      
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000000005)   .minimize(loss)   


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 
training_epochs = 100
batch_size = 100
total_batch = int(len(x_train)/batch_size)  
# 만약에 / 안맞아 떨어지면 모자란 값은 버린다.

for epoch in range(training_epochs):
    avg_loss = 0
    
    for i in range(total_batch):    # 몇번? 600번
        
        start = i * batch_size      # 0
        end = start + batch_size    # 100
        batch_x, batch_y = x_train[start:end], y_train[start:end] # 0~100
        
        feed_dict = {x:batch_x,y:batch_y}   # 단순변수
        
        batch_loss, _ = sess.run([loss, optimizer],feed_dict=feed_dict)

        avg_loss += batch_loss / total_batch
    
    print(f"Epoch : {epoch+1:04d}, loss : {avg_loss:.9f}")

print("훈련 끝")

prediction = tf.equal(tf.math.argmax(hypothesis,1), tf.math.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print(f'ACC : {sess.run(accuracy,feed_dict={x:x_test,y:y_test})}')




'''
while True:
    avg_loss = 0
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
       
        break
'''    
sess.close()