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

#2. 모델
x = tf.compat.v1.placeholder(tf.float32,shape=[None,30])    
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1]) 

w = tf.compat.v1.Variable(tf.zeros([30,10]),'weights')
b = tf.compat.v1.Variable(tf.zeros([10]))

hidden_layer1 = tf.matmul(x, w) + b

w1 = tf.compat.v1.Variable(tf.random.normal([10,20], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random.normal([20], name='bias1'))

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([20, 10], name='weight2'))
b2 = tf.compat.v1.Variable(tf.random.normal([10], name='bias2'))

hidden_layer3 = tf.matmul(hidden_layer2, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([10, 5], name='weight3'))
b3 = tf.compat.v1.Variable(tf.random.normal([5], name='bias3'))

hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([5, 1], name='weight3'))
b4 = tf.compat.v1.Variable(tf.random.normal([1], name='bias3'))

hypothesis = tf.sigmoid(tf.matmul(hidden_layer4,w4) + b4)
       
#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))   

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0000005)       
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000000005)      

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
    
    # if step == 5: break
    
    if loss_val < 0.105:        # <-- 단층의 한계를 뛰어넘었다.
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
        # train_acc는 : 0.9494505524635315 test_acc는 : 0.9385964870452881 <-- 다층으로 단층을 뛰어넘었다.
        break
    
sess.close()