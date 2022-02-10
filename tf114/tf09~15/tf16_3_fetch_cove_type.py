from sklearn.datasets import load_wine
import tensorflow as tf, numpy as np,os
from sklearn.preprocessing import OneHotEncoder            
from sklearn.metrics import mean_absolute_error,accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = load_wine()
x_data = datasets.data                      # (178, 13)
y_data = datasets.target.reshape(-1,1)      # (178,)

enco = OneHotEncoder(sparse=False)        
y_data = enco.fit_transform(y_data.reshape(-1,1))       # #  (178, 3)

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None,13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3])

w = tf.compat.v1.Variable(tf.random.normal([13,3]),'weights')
b = tf.compat.v1.Variable(tf.random.normal([1]))

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)      
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)       

#3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 

while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,optimizer],feed_dict={x:x_data,y:y_data})
    print(f"{step:05d} \t{loss_val}")
    
    if step == 5: break
    if loss_val < 0.0725:
        #4. 평가, 예측
        
        y_predict = sess.run(hypothesis, feed_dict={x:x_data})
        y_predict_int = sess.run(tf.math.argmax(y_predict,1))
        y_data_int = np.argmax(y_data,axis=1)
        acc = accuracy_score(y_data_int,y_predict_int)
        
        print(f"acc : {acc}")
        
        break
        
sess.close()