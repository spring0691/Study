import numpy as np
from sklearn.svm import LinearSVC,SVC               
from sklearn.linear_model import Perceptron         
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터  XOR
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()
# model = SVC()                                     
model = Sequential()
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))                                
model.add(Dense(1, activation='sigmoid'))  # input이 2개 output이 1개 이게 sigmoid로 activation 받아서 output이 나온다.

#3. 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data,y_data,batch_size=1, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)

results = model.evaluate(x_data,y_data)

print(x_data, "의 예측결과 : ", y_predict)
print("metrics_acc : ", results[1])

acc = accuracy_score(y_data, np.round(y_predict,0).astype(int))
print("accuracy_score : ", acc)

# Sequential 써서 Dense를 썼는데도 단층퍼셉트론으로는 Xor을 못잡는다

'''
class MultiLayer:
      def __init__(self,learning_rate=0.01,batch_size=32,hidden_node=10):
    self.w1=None #은닉층 가중치 행렬
    self.b1=None #은닉층 바이어스 배열
    self.w2=None #출력층 가중치 행렬
    self.b2=None #출력층 바이어스 배열
    self.lr=learning_rate #학습률
    self.batch_size=batch_size
    self.node=hidden_node #은닉층 노드 개수
    self.losses=[] # 손실(훈련세트)
    self.val_losses=[] # 손실(검증세트)
    self.accuracy=[] # 정확도(훈련세트)
    self.val_accuracy=[] # 정확도(검증세트)

  def init_weights(self,m,n): # 가중치 초기화 함수 
    k=self.node
    self.w1=np.random.normal(0,1,(m,k)) # 가우시안 분포로 초기화 (m x k) 행렬
    self.b1=np.zeros(k) # 0으로 초기화
    self.w2=np.random.normal(0,1,(k,n)) # 가우시안 분포로 초기화 (k x n) 행렬
    self.b2=np.zeros(n) #0으로 초기화

  def sigmoid(self,z1): #sigmoid 함수
    z1=np.clip(z1,-100,None) 
    a1=1/(1+np.exp(-z1)) 
    return a1

  def softmax(self,z2): #softmax 함수
    z2=np.clip(z2,-100,None)
    exp_z2=np.exp(z2)
    a2=exp_z2/np.sum(exp_z2,axis=1).reshape(-1,1)
    return a2

  def forward1(self,x): # 입력층 -> 은닉층 순전파
    z1=np.dot(x,self.w1)+self.b1
    return z1

  def forward2(self,a1): # 은닉층 -> 출력층 순전파
    z2=np.dot(a1,self.w2)+self.b2
    return z2

  def loss(self,x,y): # 손실 계산
    z1=self.forward1(x)
    a1=self.sigmoid(z1)
    z2=self.forward2(a1)
    a2=self.softmax(z2)
    
    a2=np.clip(a2,1e-10,1-1e-10)
    return -np.sum(y*np.log(a2)) #크로스 엔트로피

  def backpropagation(self,x,y): # 역전파 알고리즘
    z1=self.forward1(x)
    a1=self.sigmoid(z1)
    z2=self.forward2(a1)
    a2=self.softmax(z2)
  
    w2_grad=np.dot(a1.T,-(y-a2)) # 은닉층 가중치의 기울기 계산
    b2_grad=np.sum(-(y-a2)) # 은닉층 바이어스의 기울기 계산

    #은닉층으로 전파된 오차 계산
    hidden_err=np.dot(-(y-a2),self.w2.T)*a1*(1-a1)

    w1_grad=np.dot(x.T,hidden_err) # 입력층 가중치의 기울기 계산
    b1_grad=np.sum(hidden_err,axis=0) #입력층 바이어스의 기울기 계산

    return w1_grad,b1_grad,w2_grad,b2_grad

  def minibatch(self,x,y):
    iter=math.ceil(len(x)/self.batch_size)
    
    x,y=shuffle(x,y)
    
    for i in range(iter):
      start=self.batch_size*i
      end=self.batch_size*(i+1)
      yield x[start:end],y[start:end]

  def fit(self,x_data,y_data,epochs=40,x_val=None,y_val=None):
    self.init_weights(x_data.shape[1],y_data.shape[1]) # 가중치 초기화
    for epoch in range(epochs):
      l=0 # 손실 누적할 변수
      #가중치 누적할 변수들 초기화
      for x,y in self.minibatch(x_data,y_data):
      
        l+=self.loss(x,y) # 손실 누적
        # 역전파 알고리즘을 이용하여 각 층의 가중치와 바이어스 기울기 계산
        w1_grad,b1_grad,w2_grad,b2_grad=self.backpropagation(x,y)

        # 은닉층 가중치와 바이어스 업데이트  
        self.w2-=self.lr*(w2_grad)/len(x)
        self.b2-=self.lr*(b2_grad)/len(x)

        # 입력층 가중치와 바이어스 업데이트
        self.w1-=self.lr*(w1_grad)/len(x)
        self.b1-=self.lr*(b1_grad)/len(x)
      
      #검증 손실 계산
      val_loss=self.val_loss(x_val,y_val)
      
      self.losses.append(l/len(y_data))
      self.val_losses.append(val_loss)
      self.accuracy.append(self.score(x_data,y_data))
      self.val_accuracy.append(self.score(x_val,y_val))

      print(f'epoch({epoch+1}) ===> loss : {l/len(y_data):.5f} | val_loss : {val_loss:.5f}',\
            f' | accuracy : {self.score(x_data,y_data):.5f} | val_accuracy : {self.score(x_val,y_val):.5f}')

  def predict(self,x_data):
    z1=self.forward1(x_data)
    a1=self.sigmoid(z1)
    z2=self.forward2(a1)
    return np.argmax(z2,axis=1) #가장 큰 인덱스 반환

  def score(self,x_data,y_data):
    return np.mean(self.predict(x_data)==np.argmax(y_data,axis=1))

  def val_loss(self,x_val,y_val): # 검증 손실 계산
    val_loss=self.loss(x_val,y_val)
    return val_loss/len(y_val)
'''