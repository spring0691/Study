import tensorflow as tf, os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

x = [1,2,3]
y = [3,5,7]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])   # 이런식으로하면 아래에서 feed_dict에서 내용을 넣어줄수있다.
# 위에서 선언하면 아래에서 feed_dict안해줘도 된다.
x_test1 = tf.constant([4], tf.float32)
x_test2 = tf.constant([5,6], tf.float32)
x_test3 = tf.constant([7,8,9], tf.float32)

w = tf.compat.v1.Variable([0.3])
b = tf.compat.v1.Variable([1.0])

hypothesis = x*w + b

loss = tf.reduce_mean(tf.square(hypothesis - y))  
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.17)   
train = optimizer.minimize(loss) 

################################################################## 1. Session() // sess.run(변수)
#1. 세션 선언하고 sess.run(변수)로 출력하는 방식
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
step = 0
while True:
    step += 1
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b])
    
    if step % 10 == 0:
        
        print(step, loss_val, w_val, b_val) 
    
    if loss_val < 1e-5:                    
            
        predict = x_test*w_val+b_val
        # feed dict방식으로 할 경우!
        predict1 = np.round(sess.run(predict,feed_dict={x_test:[4]}),0)
        predict2 = np.round(sess.run(predict,feed_dict={x_test:[5,6]}),0)
        predict3 = np.round(sess.run(predict,feed_dict={x_test:[7,8,9]}),0)
        print(f"{predict1}\n{predict2}\n{predict3}")
        break
sess.close()



################################################################## 2. Session() // 변수.eval(session=세션)
#2. 세션 선언하고 변수.eval(session=세션)으로 출력하는 방식
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
step = 0
while True:
    step += 1
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b])
    
    if step % 10 == 0:
        
        print(step, loss_val, w_val, b_val) 
    
    if loss_val < 1e-5:                    
        
        # 위에서 선언했다면 요런식으로 해준다. 이게 더 비효율적 같기도?
        predict1 = x_test1*w_val+b_val
        predict2 = x_test2*w_val+b_val
        predict3 = x_test3*w_val+b_val
        
        predict1 = np.round(predict1.eval(session=sess),0)
        predict2 = np.round(predict2.eval(session=sess),0)
        predict3 = np.round(predict3.eval(session=sess),0)
        print(f"{predict1}\n{predict2}\n{predict3}")
        break
sess.close()



################################################################## 3. InteratctiveSession() // 변수.eval
#3. 인터렉티브세션 선언하고 변수.eval로 출력하는 방식
sess = tf.compat.v1.InteractiveSession()   
sess.run(tf.compat.v1.global_variables_initializer())
step = 0
while True:
    step += 1
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b])
    
    if step % 10 == 0:
        
        print(step, loss_val, w_val, b_val) 
    
    if loss_val < 1e-5:                    
            
        predict = x_test*w_val+b_val
        
        predict1 = np.round(predict.eval(feed_dict={x_test:[4]}),0)
        predict2 = np.round(predict.eval(feed_dict={x_test:[5,6]}),0)
        predict3 = np.round(predict.eval(feed_dict={x_test:[7,8,9]}),0)
        print(f"{predict1}\n{predict2}\n{predict3}")
        break
sess.close()