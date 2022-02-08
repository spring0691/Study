import tensorflow as tf, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
# 메모리에 placeholder를 이용하여 위치를 잡아놓는다.
# 새로운 개념의 자료형 placeholder. Tensorflow에만 존재함.

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5})) # feed_dict로 메모리에 잡아놓은 placeholder에 값을 넣어줘서
# sess.run 실행하기 직전에 값을 넘겨준다.
