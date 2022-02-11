# 이미 단층신경망에서 내가 while문으로 무한반복돌리면서 단순 데이터셋으로 도달할 수 있는
# 최소의 loss, 최적의 weight에 도달해버려서 더 이상의 발전이 없는거 같다;

from sklearn.datasets import load_diabetes
import tensorflow as tf, numpy as np,os
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = load_diabetes()
x_data = datasets.data                     # (442, 10)
y_data = datasets.target.reshape(-1,1)     # (442,)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,shuffle=True,random_state=77)

#2. 모델 
x = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w1 = tf.compat.v1.Variable(tf.random.normal([10,32]),'weights1')
b1 = tf.compat.v1.Variable(tf.random.normal([32]),'bias1')

Hidden_layer1 =  tf.nn.relu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([32,16]),'weights2')
b2 = tf.compat.v1.Variable(tf.random.normal([16]),'bias2')

Hidden_layer2 = tf.nn.relu(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([16,8]),'weights3')
b3 = tf.compat.v1.Variable(tf.random.normal([8]),'bias3')

Hidden_layer3 = tf.nn.relu(tf.matmul(Hidden_layer2,w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([8,4]),'weights4')
b4 = tf.compat.v1.Variable(tf.random.normal([4]),'bias4')

Hidden_layer4 = tf.nn.relu(tf.matmul(Hidden_layer3,w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([4,1]),'weights5')
b5 = tf.compat.v1.Variable(tf.random.normal([1]),'bias5')

hypothesis = tf.matmul(Hidden_layer4,w5) + b5

#3-1.컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))            # MSE

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00002)       
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.03)      
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
    
    if loss_val < 500:     # 단층 신경만의 한계 loss를 돌파했다.
        
        y_train_predict = sess.run(hypothesis,feed_dict={x:x_train})
        y_test_predcit = sess.run(hypothesis,feed_dict={x:x_test})
        train_r2 = r2_score(y_train,y_train_predict)
        test_r2 = r2_score(y_test,y_test_predcit)
        print(f"train_r2스코어 : {train_r2} test_r2스코어 : {test_r2}")
        # train_r2스코어 : 0.5142949103499602 test_r2스코어 : 0.5296165706130218
        # train_r2스코어 : 0.9137111327790238 test_r2스코어 : 0.9215147933721487 <-- 거의 2배가 뛰었다
        break
sess.close()