from sklearn.datasets import load_breast_cancer
import tensorflow as tf, numpy as np,os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data                     # (569, 30)
y_data = datasets.target.reshape(-1,1)     # (569, 1)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,shuffle=True,random_state=77)

y_train= np.array(y_train,dtype='float32')
y_test= np.array(y_test,dtype='float32')

x = tf.compat.v1.placeholder(tf.float32,shape=[None,30])    
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1]) 

w = tf.compat.v1.Variable(tf.zeros([30,1]),'weights')
b = tf.compat.v1.Variable(tf.zeros([1]))

#2. 모델
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)      
# hypothesis = tf.matmul(x,w) + b
    
#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))   
# loss = tf.reduce_mean(tf.square(hypothesis-y))        # MSE

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005)       
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00005)      

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 

while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,train],feed_dict={x:x_train,y:y_train}) 
    print(f"{step:05d} \t{loss_val:.7f}")
    
    if loss_val < 0.359:
        #4. 평가, 예측
        
        y_train_predict = sess.run(hypothesis,feed_dict={x:x_train})
        
        y_predict = tf.cast(y_train_predict > 0.5, dtype=tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_train,y_predict), dtype=tf.float32))  
        
        train_acc = sess.run(acc)
        
        
        
        y_test_predict = sess.run(hypothesis, feed_dict={x:x_test})
        
        y_predict = tf.cast(y_test_predict > 0.5, dtype=tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_test,y_predict), dtype=tf.float32))  
        
        test_acc = sess.run(acc)
        
        print(f"train_acc는 : {train_acc} test_acc는 : {test_acc}")
        # train_acc는 : 1.0 test_acc는 : 1.0
        # train_acc는 : 0.9054945111274719 test_acc는 : 0.9210526347160339 <-- sigmoid 방식으로 한거
        break
    
sess.close()