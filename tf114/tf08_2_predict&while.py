# 실습
# 1. [4]
# 2. [5,6]
# 3. [6,7,8]

# 위 값들을 이용해서 predict해라.
# x_test라는 placeholder를 생성해서 넣어주기.

from ast import While
import tensorflow as tf, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(77)

#1. 데이터

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)


#2. 모델구성
hypothesis = x_train * w + b           # hypothesis(가설) = y_predict

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   
train = optimizer.minimize(loss) 


#3-2. 훈련

with tf.compat.v1.Session() as sess:        
    
    sess.run(tf.compat.v1.global_variables_initializer())
    step = 0
    
    while True:
        
        step += 1
        # sess.run(train)     # 여기서 실행이 일어난다.
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
        
        if step % 20 == 0:

            print(step, loss_val, w_val, b_val) 
                  
################    실습&과제  #################
        if loss_val <= 1e-8:                    # if w_val >= 0.999:
            
            # predict = x_test*w_val+b_val
            predict = x_test*w+b                # y_predict = model.predict 구현
            
            predict1 = sess.run(predict,feed_dict={x_test:[4]})
            predict2 = sess.run(predict,feed_dict={x_test:[5,6]})
            predict3 = sess.run(predict,feed_dict={x_test:[7,8,9]})
            print(f"{predict1}\n{predict2}\n{predict3}")
            break
        