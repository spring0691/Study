import tensorflow as tf
node1 = tf.constant(2.0,tf.float32)
node2 = tf.constant(3.0,tf.float32)

# 실습 
# 덧셈 node3, 뺄셈 node4, 곱셈 node5, 나눗셈 node6

# sess = tf.Session()
sess = tf.compat.v1.Session()   # Session을 더 자세한 경로로 가져오겠다.

# node3 = node1 + node2
node3 = tf.add(node1,node2)
print(sess.run(node3))

# node4 = node1 - node2
node4 = tf.subtract(node1,node2)
print(sess.run(node4))

# node5 = node1 * node2
node5 = tf.multiply(node1,node2)
print(sess.run(node5))

# node6 = node1 / node2
node6 = tf.divide(node1,node2)
print(sess.run(node6))