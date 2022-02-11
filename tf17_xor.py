import tensorflow as tf, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])    # float32 64차이 부동소수점인데 사이즈 차이.
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.random_normal([2,1], name='weight'))