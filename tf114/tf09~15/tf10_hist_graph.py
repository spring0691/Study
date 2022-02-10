import tensorflow as tf, os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(77)

#1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]

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
    
    loss_val_list = []
    
    while True:
        
        step += 1
        # sess.run(train)     # 여기서 실행이 일어난다.
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:x_train_data, y_train:y_train_data})
        
        if step % 10 == 0:
            print(step, loss_val, w_val, b_val) 
                  
        loss_val_list.append(loss_val)
        
        if loss_val < 1e-4:                    # if w_val >= 0.999:    loss_val <= 1e-9
            
            # predict = x_test*w_val+b_val
            predict = x_test*w+b                # y_predict = model.predict 구현
            
            predict1 = np.round(sess.run(predict,feed_dict={x_test:[4]}),0)
            predict2 = np.round(sess.run(predict,feed_dict={x_test:[5,6]}),0)
            predict3 = np.round(sess.run(predict,feed_dict={x_test:[7,8,9]}),0)
            print(f"{predict1}\n{predict2}\n{predict3}")
            break

# tensor2버전에서 hist를 [ ]에 담아서 plt출력하는것과 같은 원리

import matplotlib.pyplot as plt
plt.plot(loss_val_list[100:])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()