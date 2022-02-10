import numpy as np, tensorflow as tf,os
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],      # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],      # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],      # 0
          [1,0,0]]

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,3]),name='bias')        # y의 colmun이 3개이기 때문.

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)
# model.add(Dense(3, activation='softmax))

#3-1.컴파일
# loss = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))     # binary nono~
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# categorical_crossentropy

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)      
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.6).minimize(loss)       

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 

while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,optimizer],feed_dict={x:x_data,y:y_data})
    print(f"{step:05d} \t{loss_val}")
    
    
    if loss_val < 0.0047:
        #4. 평가, 예측
        
        y_predict = sess.run(hypothesis, feed_dict={x:x_data})
        y_predict_int = sess.run(tf.math.argmax(y_predict,1))
        y_data_int = np.argmax(y_data,axis=1)
        acc = accuracy_score(y_data_int,y_predict_int)
        
        print(f"acc : {acc}")
        
        break
        
sess.close()