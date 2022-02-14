from keras.datasets import mnist
import tensorflow as tf, numpy as np,os
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)/255.      # 60000, 784
x_test = x_test.reshape(len(x_test),-1)/255.         # 10000, 784

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w = tf.compat.v1.Variable(tf.random.normal([x_train.shape[1], 64], mean=0.0, stddev=tf.math.sqrt(2/(13+64)), name='weight'))
b = tf.compat.v1.Variable(tf.zeros([64]), name='bias')

hidden_layer1 = tf.matmul(x, w) + b

w1 = tf.compat.v1.Variable(tf.random.normal([64, 32], mean=0, stddev=tf.math.sqrt(2/(64+32)), name='weight1'))
b1 = tf.compat.v1.Variable(tf.zeros([32], name='bias1'))

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([32, 16], mean=0, stddev=tf.math.sqrt(2/(32+16)), name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([16], name='bias2'))

hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([16, 4], mean=0, stddev=tf.math.sqrt(2/(16+4)), name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([4], name='bias3'))

hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([4, 10], mean=0, stddev=tf.math.sqrt(2/(4+1)), name='weight3'))
b4 = tf.compat.v1.Variable(tf.zeros([10], name='bias3'))

hypothesis = tf.nn.softmax(tf.matmul(hidden_layer4,w4) + b4)
       
#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))   # categorical_crossentropy

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)      
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000000000005)   .minimize(loss)   

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 

while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,optimizer],feed_dict={x:x_train,y:y_train})
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