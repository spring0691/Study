from sklearn.datasets import load_boston
import tensorflow as tf, numpy as np,os
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

# mlp는 각 레이어에서 다음레이어로 넘어갈때 값을 줄여서 넘겨줘야한다. 그냥 넘겨주면 의미가없다.
# sigmoid 내장, tanh 내장, 

#1. 데이터
datasets = load_boston()
x_data = datasets.data       # (506, 13)
y_data = datasets.target.reshape(-1,1)     # (506,)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,shuffle=True,random_state=77)

#2. 모델 
x = tf.compat.v1.placeholder(tf.float32, shape=[None,13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w1 = tf.compat.v1.Variable(tf.zeros([13,64]),'weights1')
b1 = tf.compat.v1.Variable(tf.zeros([64]),'bias1')

Hidden_layer1 =  (tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.zeros([64,32], name='weight2'))   
b2 = tf.compat.v1.Variable(tf.zeros([32], name='bia2'))     

Hidden_layer2 = tf.matmul(Hidden_layer1,w2) + b2

w3 = tf.compat.v1.Variable(tf.zeros([32,16], name='weight3'))   
b3 = tf.compat.v1.Variable(tf.zeros([16], name='bia3'))     

Hidden_layer3 = tf.matmul(Hidden_layer2,w3) + b3

w4 = tf.compat.v1.Variable(tf.zeros([16,8], name='weight4'))   
b4 = tf.compat.v1.Variable(tf.zeros([8], name='bia4')) 

Hidden_layer4 = tf.matmul(Hidden_layer3,w4) + b4

w5 = tf.compat.v1.Variable(tf.zeros([8,1], name='weight5'))   
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bia5')) 

hypothesis = tf.matmul(Hidden_layer4,w5) + b5

#3-1.컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))            # MSE

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)  # tf.zeros로 가중치를 0으로 시작하면 lr을 크게해도 안터진다. 
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)      
train = optimizer.minimize(loss)

#3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 
val_loss_list = []

while True:
    step += 1
    p =1000
    
    loss_val, _ = sess.run([loss, train],feed_dict={x:x_train,y:y_train})
    
    print(f"{step:05d}\t{loss_val:.7f}")
    
    val_loss_list.append(loss_val)
    
    # if len(val_loss_list) > p:
    #     if val_loss_list[-p] < val_loss_list[-p+1:-1]:   # patience값번째 뒤의 값. vs 그 뒤의 patience개수만큼의 모든 값
    
    if loss_val < 22.8558:
        
        y_train_predict = sess.run(hypothesis,feed_dict={x:x_train})
        y_test_predcit = sess.run(hypothesis,feed_dict={x:x_test})
        train_r2 = r2_score(y_train,y_train_predict)
        test_r2 = r2_score(y_test,y_test_predcit)
        print(f"train_r2스코어 : {train_r2} test_r2스코어 : {test_r2}")
        # train_r2스코어 : 0.739819088874253 test_r2스코어 : 0.7271981131877968
        break
sess.close()