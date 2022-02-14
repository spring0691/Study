from keras.datasets import mnist
import tensorflow as tf, numpy as np,os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)      # 60000, 784
x_test = x_test.reshape(len(x_test),-1)         # 10000, 784

#2. 모델
x = tf.compat.v1.placeholder(tf.float32,shape=[None,784])    
y = tf.compat.v1.placeholder(tf.float32,shape=[None,10]) 

w = tf.compat.v1.Variable(tf.zeros([784,1024]),'weights')
b = tf.compat.v1.Variable(tf.zeros([1,1024]))

hidden_layer1 = tf.matmul(x, w) + b

w1 = tf.compat.v1.Variable(tf.random.normal([1024,512], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random.normal([1,512], name='bias1'))

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([512, 256], name='weight2'))
b2 = tf.compat.v1.Variable(tf.random.normal([1,256], name='bias2'))

hidden_layer3 = tf.matmul(hidden_layer2, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([256, 128], name='weight3'))
b3 = tf.compat.v1.Variable(tf.random.normal([1,128], name='bias3'))

hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([128, 64], name='weight4'))
b4 = tf.compat.v1.Variable(tf.random.normal([1,64], name='bias4'))

hidden_layer5 = tf.nn.relu(tf.matmul(hidden_layer4, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([64, 32], name='weight5'))
b5 = tf.compat.v1.Variable(tf.random.normal([1,32], name='bias5'))

hidden_layer6 = tf.nn.relu(tf.matmul(hidden_layer5, w5) + b5)

w5 = tf.compat.v1.Variable(tf.random.normal([32, 16], name='weight5'))
b5 = tf.compat.v1.Variable(tf.random.normal([1,16], name='bias5'))

hidden_layer7 = tf.nn.relu(tf.matmul(hidden_layer6, w5) + b5)

w6 = tf.compat.v1.Variable(tf.random.normal([16, 10], name='weight6'))
b6 = tf.compat.v1.Variable(tf.random.normal([1,10], name='bias6'))

hypothesis = tf.nn.softmax(tf.matmul(hidden_layer7,w6) + b6)
       
#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))   # categorical_crossentropy

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
    
    if loss_val < 0.105:       
        #4. 평가, 예측
        
        y_train_predict = sess.run(hypothesis, feed_dict={x:x_train})
        y_train_predict_int = sess.run(tf.math.argmax(y_train_predict,1))
        y_train_int = np.argmax(y_train,axis=1)
        train_acc = accuracy_score(y_train_int,y_train_predict_int)
        
        y_test_predict = sess.run(hypothesis, feed_dict={x:x_test})
        y_test_predict_int = sess.run(tf.math.argmax(y_test_predict,1))
        y_test_int = np.argmax(y_test,axis=1)
        test_acc = accuracy_score(y_test_int,y_test_predict_int)
        
        print(f"train_acc : {train_acc} test_acc : {test_acc}")
       
        break
    
sess.close()