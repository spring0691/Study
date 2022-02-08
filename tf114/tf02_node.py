# 신경망에서 노드 구성하는걸 tf1환경에서 진행해보겠다.
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)    # tensor연산을 하기위해서 텐서플로라는걸 명시해줘야함.
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)

sess = tf.Session()                     # 이걸 꼭 써줘야 연산이 실행된다.
print('node1, node2 : ', sess.run([node1, node2]))         # 2개 이상 출력하려면 리스트로 묶어줘야한다.
print('node3 : ',sess.run(node3))
