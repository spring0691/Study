import pandas as pd, numpy as np, tensorflow as tf,os
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

path = '../Project/Kaggle_Project/bike/'
Bikedata = pd.read_csv(path + 'train.csv') 

x = Bikedata.drop(['casual','registered','count'], axis=1)  
x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)
y = Bikedata['count']  
# y = np.log1p(y)

x_data = x.values                   # (10886, 12)
y_data = y.values.reshape(-1,1)     # (10886,)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,shuffle=True,random_state=77)

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None,12])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.random.normal([12,32]),'weights')
b = tf.compat.v1.Variable(tf.random.normal([32]), 'bias')

hidden_layer1 = tf.nn.relu(tf.matmul(x, w) + b)

w1 = tf.compat.v1.Variable(tf.random.normal([32, 16], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random.normal([16], name='bias1'))

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([16, 8], name='weight2'))
b2 = tf.compat.v1.Variable(tf.random.normal([8], name='bias2'))

hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([8, 4], name='weight3'))
b3 = tf.compat.v1.Variable(tf.random.normal([4], name='bias3'))

hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([4, 1], name='weight3'))
b4 = tf.compat.v1.Variable(tf.random.normal([1], name='bias3'))

hypothesis = tf.matmul(hidden_layer4,w4) + b4

#3-1.컴파일
# loss = mean_absolute_error(y,hypothesis)              # MAE
loss = tf.reduce_mean(tf.square(hypothesis-y))          # MSE

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000001)       
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.004)      
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
    
    if loss_val < 15400:    # <-- loss값 돌파했다.
        
        y_train_predict = sess.run(hypothesis,feed_dict={x:x_train})
        y_test_predcit = sess.run(hypothesis,feed_dict={x:x_test})
        train_r2 = r2_score(y_train,y_train_predict)
        test_r2 = r2_score(y_test,y_test_predcit)
        print(f"train_r2스코어 : {train_r2} test_r2스코어 : {test_r2}")
        # train_r2스코어 : 0.34002185158992193 test_r2스코어 : 0.327923071346776
        # train_r2스코어 : 0.5313489178386734 test_r2스코어 : 0.5270370013856264 <-- 단층모델보다 성능이 향상되었다.
        break
    
    # if loss_val < 100:    # <-- loss값 돌파했다.    log넣어서했는데 결과 완전망함
        
    #     y_train_predict = sess.run(hypothesis,feed_dict={x:x_train})
    #     y_test_predcit = sess.run(hypothesis,feed_dict={x:x_test})
    #     train_r2 = r2_score(np.expm1(y_train),np.expm1(y_train_predict))
    #     test_r2 = r2_score(np.expm1(y_test),np.expm1(y_test_predcit))
    #     print(f"train_r2스코어 : {train_r2} test_r2스코어 : {test_r2}")
        # train_r2스코어 : 0.34002185158992193 test_r2스코어 : 0.327923071346776
        # train_r2스코어 : 0.5313489178386734 test_r2스코어 : 0.5270370013856264 <-- 단층모델보다 성능이 향상되었다.
        # train_r2스코어 : -8.607907041858832e+25 test_r2스코어 : -710993069626442.5 
        break
sess.close()