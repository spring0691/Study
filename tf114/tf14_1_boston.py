from sklearn.datasets import load_boston
import tensorflow as tf, numpy as np
from sklearn.metrics import r2_score,mean_absolute_error
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = load_boston()
x_data = datasets.data       # (506, 13)
y_data = datasets.target.reshape(-1,1)     # (506,)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.random.normal([13,1]),'weights')
b = tf.compat.v1.Variable(tf.random.normal([1]))

#2. 모델 
hypothesis = tf.matmul(x,w) + b

#3-1.컴파일
loss = tf.losses.mean_squared_error(y,hypothesis)       # MAE
# loss = tf.reduce_mean(tf.square(hypothesis-y))            # MSE

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000001)       
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.03)      
train = optimizer.minimize(loss)

#3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 
val_loss_list = []

while True:
    step += 1
    p =1000
    
    loss_val, _ = sess.run([loss, train],feed_dict={x:x_data,y:y_data})
    
    print(f"{step:05d}\t{loss_val:.7f}")
    
    val_loss_list.append(loss_val)
    
    # if len(val_loss_list) > p:
    #     if val_loss_list[-p] < val_loss_list[-p+1:-1]:   # patience값번째 뒤의 값. vs 그 뒤의 patience개수만큼의 모든 값
    
    if loss_val < 22:
        predict = tf.matmul(x,w) + b
        y_predict = sess.run(predict,feed_dict={x:x_data,y:y_data})
        
        r2 = r2_score(y_data,y_predict)
        
        print(f"r2스코어는 : {r2}")
        
        break
sess.close()