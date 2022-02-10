import tensorflow as tf,os,numpy as np
from sklearn.metrics import r2_score,mean_absolute_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(66)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

# x는 (5,3), y는 (5,1) 또는 (5,)
# y = x1*w1 + x2*w2 + x3*w3 + b

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# w = tf.compat.v1.Variable(-10)                                    # 직접 값 주고 싶을때
w1 = tf.compat.v1.Variable(tf.random.normal([1]), name='weight1')   # dtype=tf.float32가 default
w2 = tf.compat.v1.Variable(tf.random.normal([1]), name='weight2')
w3 = tf.compat.v1.Variable(tf.random.normal([1]), name='weight3')   # ([1])이 의미하는것 -> shape.  1개의 값을 주겠다
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')       

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run([w1,w2,w3,b]))                                     # 4개의 값은 다르지만 random_seed가 같아서 같은 값이 재현된다.          

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일
loss = tf.losses.mean_squared_error(y,hypothesis)       # mae하기 위한 시도..
# loss = tf.reduce_mean(tf.square(hypothesis-y))        # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000454)       # 1e-5 = 0.00001
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 
print(f"Step \t loss_val \t w1_val \t w2_val \t w3_val \t b_val")

while True:
    step += 1
    
    _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b],feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
    print(f"{step:05d}\t{loss_val:.7f} \t{w1_val[0]:.5f} \t{w2_val[0]:.5f} \t{w3_val[0]:.5f} \t{b_val[0]:.5f}")
    
    if loss_val <= 0.175:
        predict = x1*w1 + x2*w2 + x3*w3 + b
        y_predict = sess.run(predict,feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
        
        r2 = r2_score(y_data,y_predict)
        mae = mean_absolute_error(y_data,y_predict)
        
        print(f"r2스코어는 : {r2}")
        print(f"MAE는 : {mae}")
        
        break
sess.close()
    