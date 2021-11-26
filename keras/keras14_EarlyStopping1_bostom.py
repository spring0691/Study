# 과적합 예제

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import EarlyStopping    # 웬만한 인공지능은 이미 다 구현이 되어있다. import해서 쓰면 된다. 


#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping  #정의를 해줘야 쓸수있다. 
es = EarlyStopping(monitor="val_loss", patience=50, mode='min', verbose=1)
# 멈추는 시점은 최소값.발견 직후 100번째에서 멈춘다. 
# 단 이때 제공되는 val_loss값은 최소값일까 그 이후 patience값만큼 지나서의 val_loss값일까?
# val_loss값 비교해보고 그 외의 파라미터나 설정값? 있으면 찾아보기 

start = time.time()
hist = model.fit(x_train,y_train,epochs=10000, batch_size=10,validation_split=0.25, callbacks=[es]) 
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')
#4. 평가 , 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)


plt.figure(figsize=(9,6)) # 판 깔고 사이즈가 9,5이다.
plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 그림그렸을때 격자를 보여주게 하기 위해 , 모눈종이 역할?
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') # 그림그렸을때 나오는 설명? 정보들 표시되는 위치
plt.show()

