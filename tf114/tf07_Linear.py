# y = xw + b    행렬연산에서 x와 w의 순서는 아주 중요함.

import tensorflow as tf, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_train = [1,2,3]               # batch가 3인 상태와 같다.
y_train = [1,2,3]

w = tf.compat.v1.Variable(1, dtype=tf.float32)
# w = tf.compat.v1.constant(1, dtype=tf.float32)    # 상수로 잡으면 갱신이 안되서 실행이 안된다.
b = tf.compat.v1.Variable(1, dtype=tf.float32)
# b = tf.compat.v1.constant(1, dtype=tf.float32)

#2. 모델구성
hypothesis = x_train * w + b           # hypothesis(가설) = y_predict

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # square 제곱하다.  reduce_mean 차원의 평균을 구한다. (배열의 평균) mse
# [2,3,4] - [1,2,3] = [1,1,1] -> 제곱하면 [1,1,1] 이거의 차원평균은 (1+1+1)/3 = 1   

# loss = tf.keras.losses.MeanAbsoluteError(tf.reduce_mean(y_train - y_pred))        # Mae로 하고싶지만 실패..

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)   # 경사하강법 옵티마이저

train = optimizer.minimize(loss)    # loss값을 최소로(minimize) 잡는 방향으로 설계해라

# 이 과정이 tensor2버전부터는 model.compile(loss='mse', optimizer='sgd')한 줄로 나타내어진다. 이 얼마나 편리한가.
#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(10):
    sess.run(train)     # 여기서 실행이 일어난다.
    if step % 1 == 0:
        print(f"{step+1}, {sess.run(loss)}, {sess.run(w)}, {sess.run(b)}")

sess.close()

'''
그래프 연산 작동순서
1. x * w를 실행한다.
2. x * w의 값에 b를 더해준다. = hypothesis(y_predict)
3. hypothesis의 값과 실제 y_train의 각 배열의 차를 제곱한다.(square)
4. 3번의 square해준 값을 reduce_mean으로 배열평균을 구하고 이게 loss가 된다.
5. 이 loss를 최소화 하는 방향으로 GradientDescentOptimizer가 실행되면서 
   w,b를 갱신해간다. 이에 따라 loss는 자동으로 같이 바뀐다.(minimize 최소화)

    이 w,b를 갱신해나가는 방법이자 공식이 공식문서에는 나와 있다고 하는데 너무 복잡하고 어려워서 감히 이해할 시도를 못한다.

'''

  
'''
# reduce_mean 설명
x = tf.constant([[1., 3.], [2., 6.]])

sess = tf.Session()

print(sess.run(x))
print(sess.run(tf.reduce_mean(x)))
print(sess.run(tf.reduce_mean(x, 0)))   
print(sess.run(tf.reduce_mean(x, 1)))

[[1. 3.]
[2. 6.]]

3.0

[1.5 4.5]   행의 평균 1행과 2행의 1번째 컬럼들의 평균, 1행과 2행의 2번째 컬럼들의 2번째평균

[2. 4.]     열의 평균 1열과 2열의 첫번째들의 평균, 두번째들의 평균
'''