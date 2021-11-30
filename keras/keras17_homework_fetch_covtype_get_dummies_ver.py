from sklearn.datasets import fetch_covtype              
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from pandas import get_dummies

#1. 데이터 
datasets = fetch_covtype()
                                                    
x = datasets.data   
y = datasets.target

y = get_dummies(y)                  # 이런식으로 변환해주고
#print(y.shape)                     # (581012, 7)
#print(y)                           # 출력해서 확인해보면 변환 잘 되어있고 [581012 rows x 7 columns] 행의 개수와 유니크값까지 다 표시해서 보기좋게 정리해서 보여준다.
                                    # index와 colums정보가 들어가있다고 보면 된다.
 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)   


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
