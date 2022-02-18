import numpy as np, tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
image = np.array([[[[1], [2], [3]],
                    [[4], [5], [6]],
                    [[7], [8], [9]]]])

print(image.shape)

x = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 1])
w = tf.constant([[[[1.]], [[1.]]],
                  [[[1.]], [[1.]]]])

print(w)  # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)

L1 = tf.nn.conv2d(x, w, strides=(1, 1, 1, 1), padding='VALID')

print(L1)  # Tensor("Conv2D:0", shape=(?, 2, 2, 1), dtype=float32)

sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={x:image})  # 그냥 곱해준다 행렬곱 x 

print(output, "\n", output.shape)