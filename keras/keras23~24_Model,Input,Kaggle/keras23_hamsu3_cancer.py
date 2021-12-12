######### model = Sequential()  -> model = Model

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 미리 처리한다 -> 전처리
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical 

#1.데이터 로드 및 정제

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

y = to_categorical(y)
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)    # train과 test로 나누고나서 스케일링한다.

##################################### 스케일러 설정 옵션 ########################################
scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler() 
x_train = scaler.fit_transform(x_train)    
x_test = scaler.transform(x_test) 


#2. 모델구성,모델링

input1 = Input(shape=(30,))
dense1 = Dense(50)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(15,activation="relu")(dense2)
dense4 = Dense(8,activation="relu")(dense3)
dense5 = Dense(5)(dense4)
output1 = Dense(2, activation='sigmoid')(dense5)
model = Model(inputs=input1,outputs=output1)

# model = Sequential()
# model.add(Dense(50, input_dim=30))
# model.add(Dense(30))
# model.add(Dense(15,activation="relu")) #
# model.add(Dense(8,activation="relu")) #
# model.add(Dense(5))
# model.add(Dense(2, activation='sigmoid'))
#model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) 
es = EarlyStopping  
es = EarlyStopping(monitor="val_accuracy", patience=50, mode='max',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=10,validation_split=0.111111, callbacks=[es])

model.save("./_save/keras25_3_save_cancer.h5")
#model = load_model("./_save/keras25_3_save_cancer.h5")

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

