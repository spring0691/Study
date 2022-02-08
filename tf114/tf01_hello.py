import tensorflow as tf
print(tf.__version__)

# print('hello world')

hello = tf.constant("Hello World")  # constant 상수. 고정된 값.
print(hello)                        
a = 22                              # 변수. 항상변하는 값. 언제든 다른 값을 쓸 수 있다.
AAA = 22                            # 이렇게 쓰면 상수로 정의 된다.

# tf.constant   tf.variable   tf.placeholder

# tensorflow는 그래프연산이다.

# constant에 Hello World를 집어넣고 print하면 그 정보가 나오고 Hello World가 나오진 않는다.
# 텐서플로에서 Hello World라는 output을 출력하려면 텐서머신을 지나야 하는데
# input에 데이터를 집어넣고 sess.run(op)를 통과하면 계산을 하여 output이 출력된다.
# 이게 그래프연산이고 텐서플로1의 작동방식이다...   

# 우리가 이제 어떤 로직을 만들고 그걸 돌려보고 output을 뽑고싶다면 텐서1에서는
# 텐서머신을 사용하여야만 원하는 로직을 구동시켜볼수있다.   sess.run(op)    

sess = tf.Session()             # sess.run을 만든다는 개념??
print(sess.run(hello))          # b'Hello World'으로 출력된다.
print(tf.Session().run(hello))  # 이게 풀어서쓰는 경우.