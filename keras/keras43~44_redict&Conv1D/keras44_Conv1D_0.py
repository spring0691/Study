# RNN방식을 한단계 더 보완해주기위해 양방향으로 순환시켜서 더 좋은 성능 향상을 기대해본다.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])     

y = np.array([4,5,6,7])                             

#print(x.shape, y.shape)     # (4,3) (4,)

x = x.reshape(4,3,1)    

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(10,input_shape=(3,1), return_sequences=True)) 
#model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1)))   
model.add(Conv1D(10,2,input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(10))        # Dense는 3차도 입력받는다. Dense는 무조건 그대로. 근데 위에서 flatten 해주는게 좋다.                                         
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))    
                     
#model.summary()


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  

#4. 평가, 예측

model.evaluate(x,y)
y_pred = np.array([5,6,7]).reshape(1,3,1)
result = model.predict(y_pred)   
print(result)

