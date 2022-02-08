import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())     # 즉시 실행   True

tf.compat.v1.disable_eager_execution()      # 즉시 실행을 꺼라
# Tensor2에서 이 즉시실행 기능을 끄고 tensor1문법을 쓰면 Tensor1방식으로 실행할수있다.

print(tf.executing_eagerly())     # False

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))