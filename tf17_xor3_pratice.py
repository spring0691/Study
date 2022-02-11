### [실습] 히든레이어를 2개 이상으로 늘려라!!!

import tensorflow as tf, os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

#2. 모델구성
# Input Layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])    # float32 64차이 부동소수점인데 사이즈 차이.
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

# None,2 -> None,3 -> None,1로 나가는 다층 퍼셉트론을 만들거다.

w1 = tf.compat.v1.Variable(tf.random.normal([2,2], name='weight1'))   
b1 = tf.compat.v1.Variable(tf.random.normal([2], name='bias1'))      

Hidden_layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)
# Hidden_layer1 = tf.matmul(x,w1) +b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) +b1)

w2 = tf.compat.v1.Variable(tf.random.normal([2,2], name='weight2'))   
b2 = tf.compat.v1.Variable(tf.random.normal([2], name='bia2'))      

Hidden_layer2 = tf.sigmoid(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([2,1], name='weight3'))   
b3 = tf.compat.v1.Variable(tf.random.normal([1], name='bia3'))      

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer2,w3) + b3)
# 위에서 아래로 계속 내려오면서 각 레이어마다 input output을 서로 연결하면서 받는다. 그리고 각 레이어마다 weight가 있다.

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))   
# loss = tf.reduce_mean(tf.square(hypothesis-y))        # MSE

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.06)       
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.5)      

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 

while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,train],feed_dict={x:x_data,y:y_data})
    print(f"{step:05d} \t{loss_val:.7f}")
    
    
    if loss_val < 0.00001:       #  멀티레이어 문제로 Xor를 해결. loss값이 극한으로 작아져간다. 0.0000100
        #4. 평가, 예측          
        
        y_data_predict = sess.run(hypothesis,feed_dict={x:x_data})
        print(y_data_predict)
        y_data_predict_boolean = tf.cast(hypothesis > 0.5, dtype=tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.equal(y,y_data_predict_boolean), dtype=tf.float32))
        
        pred, acc = sess.run([y_data_predict_boolean,acc],feed_dict={x:x_data,y:y_data})

        print("===========================")
        print(f"예측값 : {pred}")
        print(f"acc는 : {acc}")
        
        break
    
sess.close()