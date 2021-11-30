from sklearn.datasets import fetch_covtype              
from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical      


#1. 데이터 
datasets = fetch_covtype()
                                                    
x = datasets.data   
y = datasets.target

y = to_categorical(y)
#print(y.shape) (581012,8)
#print(y[:100]) 확인해보면 [1,0,0,0,0,0,0,0] ~ [0,0,0,0,0,0,0,1]까지 0~7의 값을 주는데 0 자체를 안쓰는 경우라고 보면되겠다.
#비효율적이다 [0,1,0,0,0,0,0,0]부터 1이라고 해석하면 되겠다
#print(y)   y값엔 그냥 변환된 값만 들어있다.
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)    # 여기는 뭐 모르면 공부 접어야지


#2. 모델링 모델구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=54))    
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))   
model.add(Dense(10))
model.add(Dense(8, activation='softmax'))   

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=1000000,batch_size=100, verbose=1,validation_split=0.2, callbacks=[es])    

#4. 평가
loss = model.evaluate(x_test,y_test)   
print('loss값과 accuracy값 : ', loss)    

#5. 예측     비교해보면 정확도를 검증할수 있다
results = model.predict(x_test[:10])
print(y_test[:10])
print(results)
