from sklearn.datasets import load_iris
import tensorflow as tf, numpy as np,os
from sklearn.preprocessing import OneHotEncoder            
from sklearn.metrics import mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = load_iris()
x_data = datasets.data              # (150, 4)
y_data = datasets.target            # (150, 1)

# sess = tf.compat.v1.Session()
# y_data = sess.run(tf.one_hot(y_data, depth=3))        tf1방식의 원핫인코딩

enco = OneHotEncoder(sparse=False)        
y_data = enco.fit_transform(y_data.reshape(-1,1))       # #  (150,3)

x_train,x_test,y_train,y_test = train_test_split(
    x_data,y_data,train_size=0.8,shuffle=True,random_state=77,stratify=y_data)

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3])

w = tf.compat.v1.Variable(tf.random.normal([4,3]),'weights')
b = tf.compat.v1.Variable(tf.random.normal([1,3]))

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)      
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.6).minimize(loss)       

#3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 

while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,optimizer],feed_dict={x:x_train,y:y_train})
    print(f"{step:05d} \t{loss_val}")
    
    
    if loss_val < 0.0484:
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
        # train_acc : 0.9833333333333333 test_acc : 1.0
        break
        
sess.close()