# mulit_variable

import tensorflow as tf,os,numpy as np
from sklearn.metrics import r2_score,mean_absolute_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]                         # (5,3)

y_data = [[152],[185],[180],[196],[142]]        # (5,1)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,3])     # column이 3개인것정보만 가져오고 행개수 무의미.
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])     

w = tf.compat.v1.Variable(tf.random.normal([3,1]),'weights')    # weight의 행 부분은 x의 열의 개수. weight의 열 부분은 y의 열의 개수     
# x*w = y가 shape이 맞아야한다. 행렬연산! (5,3) * weight = (5,1)로 나와야한다. 행렬곱은 앞의 행의열들과 뒤의 열의행들이 곱해진다.
# mv1과 비교해봤을때 mv1에서 x1_data,x2_data,x3_data와 각 w1,w2,w3를 곱해서 연산하던걸 지금 mv2파일에서 행렬연산으로 
# 구현하고 있다. x1,x2,x3를 합쳐서 (5,3)의 행렬로 만들었고 그걸 w와 곱해주는데 w도 w1,w2,w3의 합이 될 것이므로 (3,1)이 되는게 타당하다.
# x의 (5,3)에서 5는 행의 개수이므로 중요하지않고 실제로 앞의 mv1에서도 hypothesis 공식 짜는데 5개는 전혀 사용되지않았다.
# 단순 곱셈식을 행렬 연산으로 변환할때 중요한건 column의 개수이다 여기에 집중하면 헷갈릴 일이 없다. 
b = tf.compat.v1.Variable(tf.random.normal([1]))

#2. 모델
# hypothesis = x * w + b      
hypothesis = tf.matmul(x,w) + b     # 모델링에서 일반곱 말고 행렬곱을 명시해줘야 모델이 돌아간다.

#3-1. 컴파일
# loss = tf.losses.mean_squared_error(y,hypothesis)       # mae하기 위한 시도..
loss = tf.reduce_mean(tf.square(hypothesis-y))        # mse
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00004544)       # 1e-5 = 0.00001
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)       # 1e-5 = 0.00001
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 
val_loss_list = []
while True:
    step += 1
    p =1000
    
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],feed_dict={x:x_data,y:y_data})
    print(f"{step:05d}\t{loss_val:.7f}\t {w_val[0][0]:.5f}\t {w_val[1][0]:.5f}\t {w_val[2][0]:.5f}\t {b_val[0]:.5f}")
    
    val_loss_list.append(loss_val)
    
    if len(val_loss_list) > p:
        if val_loss_list[-p] < val_loss_list[-p+1:-1]:   # patience값번째 뒤의 값. vs 그 뒤의 patience개수만큼의 모든 값
            predict = tf.matmul(x,w) + b
            y_predict = sess.run(predict,feed_dict={x:x_data,y:y_data})
            
            r2 = r2_score(y_data,y_predict)
            mae = mean_absolute_error(y_data,y_predict)
            print("loss비교")
            
            print(f"r2스코어는 : {r2}")
            print(f"MAE는 : {mae}")
            
            break
sess.close()

# 무한반복 돌리고 조건문으로 끊을때의 단점. 
# 각 데이터셋 별로 최소의 loss한계값은 정해져있는데 그 값을 우리가 모르는 상태에서 프로그램한테 조건문을 넣기가 힘들다. 
# Earlystopping방식으로 모든 val_loss를 저장하고 1000번 전후의 val_loss를 비교하는 방식을 하면
# 최소의 loss값을 몰라도 자동으로 정지시킬수가있다.