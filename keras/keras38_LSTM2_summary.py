import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

#x = np.array(1,2,3,4,5,6,7)
x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])     

y = np.array([5,6,7,8])                             
        
x = x.reshape(4,1,4)

#2. 모델구성
model = Sequential()
model.add(LSTM(5,input_shape=(1,4))) 
model.add(Dense(9))        
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         

model.summary()

'''
#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  

#4. 평가, 예측

model.evaluate(x,y)
y_pred = np.array([5,6,7,8]).reshape(1,2,2)
result = model.predict(y_pred)  
print(result)

# LSTM이 뭐인지 찾아보자.
'''