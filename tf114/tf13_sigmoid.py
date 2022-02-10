import tensorflow as tf,os,numpy as np
from sklearn.metrics import accuracy_score,mean_absolute_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터 
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]          # (6,2)
y_data = [[0],[0],[0],[1],[1],[1]]                      # (6,1)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])    
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1]) 

w = tf.compat.v1.Variable(tf.random.normal([2,1]),'weights')
b = tf.compat.v1.Variable(tf.random.normal([1]))

#2. 모델
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)          # 여기서 나온 값을 sigmoid로 통과시키겠다.
# model.add(Dense(1, activation='sigmoid'))          # 이게 딥러닝방식.

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))     # binary_crossentropy 구현식 난이도 실화?

# loss 설명
# 각각의 입력이 둘 중 하나의 class를 가지는 경우로 예를 들어보겠다. 모델은 입력을 가장 잘 묘사하고 있는 class 하나를 골라야 한다. 
# 만약에 ground-truth probabailites가 y=(1.0,0.0)T인데, 모델의 예측(prediction)이 y^=(0.4,0.6)T이었다면 파라미터는 y^가 좀 더 
# y에 가까운 값을 갖을 수 있도록 조정되어야 할 것이다.
#여기서 ‘가까운’을 판단하는 척도, 다르게 말하자면 y가 y^와 얼마나 다른지 판단하는 방법이 필요하게 된다.

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.4)       
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5)      

train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

step = 0 
val_loss_list = []
while True:
    step += 1
    p =1000
    
    b_val,loss_val,w_val,hypothesis_val,_ = sess.run([b,loss, w ,hypothesis, train],feed_dict={x:x_data,y:y_data})
    print(f"{step:05d} \t{loss_val:.7f}\t{hypothesis_val[0][0]:.5f}\t{hypothesis_val[1][0]:.5f}\t{hypothesis_val[2][0]:.5f}\t"
          f"{hypothesis_val[3][0]:.5f}\t {hypothesis_val[4][0]:.5f}\t {hypothesis_val[5][0]:.5f}\t")
    
    val_loss_list.append(loss_val)
    
    # if len(val_loss_list) > p:
    #     if val_loss_list[-p] < val_loss_list[-p+1:-1]:   # patience값번째 뒤의 값. vs 그 뒤의 patience개수만큼의 모든 값
    if loss_val < 0.01:
        #4. 평가, 예측
        
        # predict = tf.sigmoid(tf.matmul(x,w) + b)
        # y_predict = sess.run(predict,feed_dict={x:x_data,y:y_data})
        # y_predict_int = np.round(y_predict,0)
        
        # tf.cast 텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.
        # Boolean형태인 경우 True이면 1, False이면 0을 출력한다. 참 거짓 판단해줌
        
        # y_predict = sess.run(tf.cast(hypothesis > 0.5, dtype=tf.float32),feed_dict={x:x_data,y:y_data})
        
        y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
        acc = tf.reduce_mean(tf.cast(tf.equal(y,y_predict), dtype=tf.float32))             
        # acc = tf.cast([y[0]!=y_predict[0],y[1]!=y_predict[1],y[2]!=y_predict[2],
                                    #   y[3]!=y_predict[3],y[4]!=y_predict[4],y[5]!=y_predict[5]], dtype=tf.float32)             
        
        # tf.equal 같은가 비교해줌  tf.equal(y,y_predict) tf.equal은 행렬을 각 행마다 비교해서 true,false를 던져줌. 앞에 tf가 텐서.행렬.
                                                                        
        pred, acc = sess.run([y_predict,acc],feed_dict={x:x_data,y:y_data})

        print("===========================")
        print("예측값 : ",hypothesis_val)
        print("예측결과 : ",pred)
        print(f"acc는 : {acc}")
        
        break
sess.close()