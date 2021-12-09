from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

#1.데이터 로드 및 정제

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
#scaler.fit(x_train)       
#x_train = scaler.transform(x_train)  
#x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
model = Sequential()
model.add(Dense(30, input_dim=30))    
model.add(Dense(25 ,activation='relu')) #   
model.add(Dense(15 ,activation='relu')) #
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2, activation='sigmoid'))
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 30)                930
_________________________________________________________________
dense_1 (Dense)              (None, 25)                775
_________________________________________________________________
dense_2 (Dense)              (None, 15)                390
_________________________________________________________________
dense_3 (Dense)              (None, 10)                160
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 6
=================================================================
Total params: 2,316
Trainable params: 2,316
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) 
es = EarlyStopping  
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=5,validation_split=0.1111111, callbacks=[es])


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

'''
결과정리                일반레이어                      relu

안하고 한 결과 
loss :              0.10503589361906052         0.09570938348770142
accuracy :          0.9298245906829834          0.9473684430122375

MinMax
loss :              0.15491396188735962         0.2635115385055542
accuracy :          0.9649122953414917          0.9298245906829834

Standard
loss :              0.1441478282213211          0.1753963828086853
accuracy :          0.9122806787490845          0.9473684430122375

Robust
loss :              0.1427469253540039          0.19655252993106842
accuracy :          0.9298245906829834          0.9122806787490845

MaxAbs
loss :              0.18201416730880737         0.27999693155288696
accuracy :          0.9298245906829834          0.9298245906829834
''' 