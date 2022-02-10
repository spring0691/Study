import tensorflow as tf,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

변수 = tf.compat.v1.Variable(tf.random.normal([1]),name='weight') 
# 출력해보면 <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref> 
# shape=(1,)의 의미는 스칼라가 1개라는 의미.
print(변수)

#1.
sess = tf.compat.v1.Session()                           # 세션 선언
sess.run(tf.compat.v1.global_variables_initializer())   # 파일 전역의 모든 변수 초기화
aaa = sess.run(변수)                                    # tf형을 출력하기위해 실행
print("aaa : ", aaa)                                    # random_seed를 66으로 고정해놔서 고정값 [0.06524777] 출력
sess.close()

#2. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)                           # 변수명.eval(session=선언한세션명)     이런 방법으로도 실행시킬수있다
print("bbb : ", bbb)
sess.close()

#3.
sess = tf.compat.v1.InteractiveSession()                # 뒤에서 작업할때 세션명시를 안해줘도 세션이 적용된다.
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print("ccc : ", ccc)
sess.close()