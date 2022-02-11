import tensorflow as tf, os, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

#2. 모델구성
# Input Layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])    # float32 64차이 부동소수점인데 사이즈 차이.
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

# None,2 -> None,3 -> None,1로 나가는 다층 퍼셉트론을 만들거다.

w1 = tf.compat.v1.Variable(tf.random.normal([2,3], name='weight1'))   
b1 = tf.compat.v1.Variable(tf.random.normal([3], name='bias1'))      

# random_normal  정규분포   random_uniform 균등분포
# numpy.random.normal(loc=0.0,scales=1.0,size=None)
# 차례대로 평균의 위치는 어디에 놓을지 여기선 0에 놓는다고 기본값으로 되어있네요.
# scales는 표준편차,size는 샘플의 사이즈를 의미하는 듯합니다.
# 정규분포로부터 임의의 샘플들을 그린다.
# (정규분포의 확률밀도함수는 가우스와 라플라스에 의해 도출된다?)

# 이게 하이퍼파라미터 튜닝 개념이다   최적의 모델을 선택해서 쓰자.
Hidden_layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)
# Hidden_layer1 = tf.matmul(x,w1) +b1
# Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) +b1)

w2 = tf.compat.v1.Variable(tf.random.normal([3,1], name='weight2'))   
b2 = tf.compat.v1.Variable(tf.random.normal([1], name='bia2'))      

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer1,w2) + b2)

# 위에서 아래로 계속 내려오면서 각 레이어마다 input output을 서로 연결하면서 받는다. 그리고 각 레이어마다 weight가 있다.

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))   
# loss = tf.reduce_mean(tf.square(hypothesis-y))        # MSE

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.06)       
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.7)      

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 

while True:
    step += 1
    p =1000
    
    loss_val,_ = sess.run([loss,train],feed_dict={x:x_data,y:y_data})
    print(f"{step:05d} \t{loss_val:.7f}")
    
    
    if loss_val < 0.0001:       #  멀티레이어 문제로 Xor를 해결. loss값이 극한으로 작아져간다.
        #4. 평가, 예측
        
        y_data_predict = sess.run(hypothesis,feed_dict={x:x_data})
        print(y_data_predict)
        y_data_predict_boolean = tf.cast(hypothesis > 0.5, dtype=tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.equal(y,y_data_predict_boolean), dtype=tf.float32))
        
        pred, acc = sess.run([y_data_predict_boolean,acc],feed_dict={x:x_data,y:y_data})

        print("===========================")
        print(f"예측값 : {pred}")
        print(f"acc는 : {acc}")
        
        break
    
sess.close()