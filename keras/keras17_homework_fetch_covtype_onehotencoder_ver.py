from sklearn.datasets import fetch_covtype    
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping    
from sklearn.preprocessing import OneHotEncoder


#1. 데이터 
datasets = fetch_covtype()
                                                    
x = datasets.data   
y = datasets.target

enco = OneHotEncoder(sparse=False)         # sparse=True가 디폴트이며 이는 Matrix를 반환한다. 원핫인코딩에서 필요한 것은 array이므로 sparse 옵션에 False를 넣어준다.
y = enco.fit_transform(y.reshape(-1,1))    # -1,1이 뭘 의미하는거지?

#print(y.shape)  # 바뀐거확인.              # (581012, 7)로 잘 바뀐걸 확인 할 수 있다  
#print(y[:10])                             # 츨력해보면 y값에 그냥 변환된값만 들어있다.
 
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
model.add(Dense(7, activation='softmax'))   

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
