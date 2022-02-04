#0.내가쓸 기능들 import
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation,Flatten, Dropout, MaxPool2D, Bidirectional, Conv1D, Conv2D, Input, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd , numpy as np
import time

# 개별 import

#1.데이터로드 및 정제

### 1-1.로드영역    데이터 형태를 x,y로 정의해주세요.


'''
### 1-2. 차원변환하기위해 shape확인.

#x값 관측.    x의 shape를 기록해주세요.     :
#print(x.shape)      

#y값 관측.    y의 shape를 기록해주세요.     :
#print(y.shape)

#모델 판별 단계.    y값 관측후 기록 및 판단  : 
#numpy      
#print(np.unique(y,return_counts=True))       

#pandas   
#print(y.value_counts())

#분류모델인경우 onehotencoding.
#from pandas import get_dummies
#y = get_dummies(y)
#from sklearn.preprocessing import OneHotEncoder
#y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))


### 1-3. 상관관계 분석 후 x칼럼제거.        스킵 가능.------------------------------------------------
#데이터가 np일 경우 pandas import해서 변환후 작업. 원핫인코딩 끄고 작업 후 다시 원핫인코딩해주세요.
# import pandas as pd
# x = pd.DataFrame(x, columns=datasets.feature_names)
# x['ydata'] = y
# #print(x.corr())
# x = x.drop(['','ydata'],axis=1)  # drop시킬 column명 기재.
# #print(x.shape)            # 변경된 칼럼개수 확인.  기재 : 
# #그 이후의 작업 계속해주기 위해 numpy로 변환
# x = x.to_numpy()
#---------------------------------------------------------------------------------------------------


### 1-4. x의 shape변환  모델링 첫 단계의 

# DNN 사용시    2차원으로 변환
#x = x.reshape(len(x),-1)

# CNN 사용시    4차원으로 변환      
x = x.reshape(len(x), , , )       #len(x)뒤의 영역은 사용자 지정입니다! 

# RNN 사용시    3차원으로 변환
#x = x.reshape(len(x), , )       #len(x)뒤의 영역은 사용자 지정입니다!   


### 1-5. train & test분리    무조건 train & test 하고 scaler.
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)


### 1-6. scaler적용. 스킵 가능----------------------------------------------------------------------

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()MinMaxScaler()

# RNN & CNN 사용시 
# 데이터를 2차원으로 만들어서 스케일링 적용하고 다시 원래차원으로 복원해줌.
#x_ train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
#x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

# DNN사용시
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)

#----------------------------------------------------------------------------------------------------

#********************* file_load 사용시 #2. #3. 전부 주석 걸어주시면 됩니다.     *******************
#model = load_model("./_save/keras25_3_save_model.h5")
#************************************************************************************************

#2.모델링   각 데이터에 알맞게 튜닝

model = Sequential()

#CNN
model.add(Conv1D~~~)
model.add(Conv2D(10,kernel_size=( 2 , 2 ), strides=1 ,padding='same',input_shape=
                    (x_train.shape[1],x_train.shape[2],x_train.shape[3]) )) # ,activation='relu'    #다음레이어의 input_shape계산 :                 
model.add(MaxPooling2D( , ))                                                                        #다음레이어의 input_shape계산 :  
model.add(Conv2D(10,kernel_size=( 2 , 2 ), strides=1, padding='same', activation='relu'))           #다음레이어의 input_shape계산 :
model.add(Flatten())

#RNN
#model.add(Bidirectional(SimpleRNN(10,input_shape=(x_train.shape[1],x_train.shape[2])   ,return_sequences=True)))       # 공백안에 ,activation='relu'도 사용해보세요. 
model.add(LSTM(10,return_sequences=True,activation='relu'))                                                             # 윗줄을 주석하고 input shape 넣지않고 바로 실행해도 알아서 모델이 돌아갑니다.
model.add(GRU(10,return_sequences=False,activation='relu'))                                                             # 다른줄을 주석처리해서 1개의 RNN만 사용해보세요


#DNN
#model.add(Dense(128,input_dim= x_train.shape[1]))                                                        # DNN방식만 적용시 위의 RNN주석 걸고 위의 1-4에서 두번째 옵션 선택합니다.                
model.add(Dense(64))                                                                                     # DNN방식 사용시 model.add(Dropout(0.5)) 복사후 사용.
model.add(Dense(32))
model.add(Dense(16,activation="relu")) #
model.add(Dense(8,activation="relu")) #
model.add(Dense(4))
model.add(Dense(,activation = ''))    # default = 'linear' 이진분류 = 'sigmoid' , 다중분류 = 'softmax' 

# compile하기전에 summary() 확인하고 파라미터확인하고 조절하고 가자
model.summary()

#3.컴파일,훈련

#**시간 출력 옵션**
#start = time.time()

model.compile(loss='', optimizer='adam')    # 회귀모델 = mse, 이진분류 = binary_crossentropy, 다중분류 = categorical_crossentropy, 분류는 ,metrics=['accuracy']
es = EarlyStopping(monitor="", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)  # monitor값 입력하세요
model.fit(x_train,y_train, epochs=10000, batch_size=10,validation_split=0.2,verbose=1,callbacks=[es])        # batch_size 센스껏 조절  

#end = time.time() - start
#print("********************** 시간 출력 *************************")
#print("걸린시간 : ", round(end, 3), '초')


#4.평가,예측        회귀모델은 r2,  분류모델은 accuracy

loss = model.evaluate(x_test,y_test)

###분류모델일때 주석 해제.
# print("----------------------loss & accuracy-------------------------")
# y_predict = model.predict(x_test)
# print(round(loss[0],4))
# print(round(loss[1],4))
# print(round(f1,4))

### 회귀모델일때 주석 해제.
# print("----------------------loss값-------------------------")
# print(round(loss,4))

# print("=====================r2score & f1score=========================")
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# f1 = f1_score(y_test,y_predict)
# print(round(r2,4))
# print(round(f1,4))


#***************************** file_save ******************************

name = import뭐시기 해서 현재 파일의 이름에서 특정값을 가져옴.
model.save(f"경로 + {파일이름따와서}.h5")

#***********************************************************************

#5.결과 정리 창

#                   DNN                 |             CNN                |               RNN
#loss:                                                                     
#                                                                    
#               DNN + Sc.                           CNN + Sc                        RNN + Sc                              
#loss:                                                                                      
#                                                                                      
#              
'''
