##########################################################
# 각각의 Scaler의 특성과 정의 정리하기
##########################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 미리 처리한다 -> 전처리
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np

#1.데이터 로드 및 정제

datasets = load_boston()
x = datasets.data
y = datasets.target

#print(x.shape)  x형태  (506, 13)
#print(y.shape)  y형태  (506,)
#print(y)        회귀모델 분류모델 확인.  이건 회귀모델

#print(np.min(x),np.max(x))      #최소값과 최대값 확인. 0.0 711.0

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)    # train과 test로 나누고나서 스케일링한다.

##################################### 스케일러 설정 옵션 ########################################
scaler = MinMaxScaler()   #어떤 스케일러 사용할건지 정의부터 해준다.
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x_train)       #어떤 비율로 변환할지 계산해줌.   여기서 구한비율로 transform해준다.
x_train = scaler.transform(x_train)   # 훈련할 데이터 변환  
x_test = scaler.transform(x_test)    # test할 데이터도 비율로 변환. 설령 스케일링 밖의 값을 받아도 이미 weight구했으므로 예측값이나온다.
##################################### 스케일러 설정 옵션 ########################################

#  StandardScaler
# 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
# 따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.

#  MinMaxScaler
# 모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
# 즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다.

#  MaxAbsScaler
# 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

#  RobustScaler
# 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.
# IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룬다.

#2. 모델구성,모델링
model = Sequential()
model.add(Dense(50, input_dim=13))
model.add(Dense(30))
model.add(Dense(15,activation="relu")) #
model.add(Dense(8,activation="relu")) #
model.add(Dense(5))
model.add(Dense(1))
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 50)                700
_________________________________________________________________
dense_1 (Dense)              (None, 30)                1530
_________________________________________________________________
dense_2 (Dense)              (None, 15)                465
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 128
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 45
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 6
=================================================================
Total params: 2,874
Trainable params: 2,874
Non-trainable params: 0
_________________________________________________________________
'''
#회귀모델 activation = linear (default값) 이진분류 sigmoid 다중분류 softmax
#회귀모델 loss = mse    이진분류 binary_crossentropy    다중분류categorical_crossentropy

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping  
es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=10,validation_split=0.111111, callbacks=[es])


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)


'''
결과정리            일반레이어                  relu추가                                          

안하고 한 결과                             
loss :            31.4710                   27.8730
r2   :            0.6822831693844669        0.7186065480994359

MinMax
loss :            28.5702                   23.6224
r2   :            0.7115684464061318        0.7615192701093052

Standard
loss :            31.4710                   13.9806
r2   :            0.6822826358440421        0.8588578155492782

Robust
loss :            31.3751                   16.9108
r2   :            0.6832512364127951        0.8292758037688225

MaxAbs
loss :            30.5554                   23.8010
r2   :            0.6915265579923042        0.7597155128379716
'''