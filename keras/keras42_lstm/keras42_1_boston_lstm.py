# 지금까지 작업한 11개 데이터들 RNN방식적용해서 모델링 해보기

#0.내가쓸 기능들 import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping


#1.데이터로드 및 정제

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)

#2.모델링
model = Sequential()
model.add(LSTM(10,input_shape=(),return_sequences=False))
model.add(20)
model.add(1,activation='')


#3.컴파일,훈련
model.compile(loss='', optimizer='adam') 
es = EarlyStopping(monitor="", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, verbose=1,callbacks=[es]) 

#4.평가,예측

model.evaluate(x_test,y_test)



