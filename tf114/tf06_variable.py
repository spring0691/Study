import tensorflow as tf, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)

# tensorflow의 모든 변수는 반드시 초기화를 해줘야한다.

init = tf.compat.v1.global_variables_initializer()
# 변수들을 초기화 시켜준다 모든 전역에.
sess.run(init)  # sess.run()을 이용해 init를 실행시켜주고.

print(sess.run(x))

# x = tf.variables_initializer([x])