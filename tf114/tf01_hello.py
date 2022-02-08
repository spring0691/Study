import tensorflow as tf
print(tf.__version__)

# print('hello world')

hello = tf.constant("Hello World")  # constant 상수. 고정된 값.
print(hello)                        
a = 22                              # 변수. 항상변하는 값. 언제든 다른 값을 쓸 수 있다.
AAA = 22                            # 이렇게 쓰면 상수로 정의 된다.