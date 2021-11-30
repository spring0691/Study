#과제1. 중위값과 평균값의 차이 비교 분석  데이터를 쭉 나열했을대 그 중 중간에 위치하는 값들의 평균 -> 분포도를 확인할수 있다.
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))

#1. 데이터 로드 및 정제
path = "./_data/bike/"   #문자열 스트링데이터    .현재폴더 ..이전폴더로 가겠다

train = pd.read_csv(path + 'train.csv')                 #print(train)    #[10886 rows x 12 columns]
test_file = pd.read_csv(path + 'test.csv')                   #print(test)     #[6493 rows x 9 columns]
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     #print(submit)    #[6493 rows x 2 columns]

print(submit_file.columns)   # ['datetime','count']



#첫번째 문제 datetime 시간 이걸 어떻게 수치화해서 데이터로 표현할것이냐
#print(train.info())     #info() 하면 데이터의 정보를 볼수있다. 객체 ? 모든 자료형의 상위 자료형
#객체(Object)란 물리적으로 존재하거나 추상적으로 생각할 수 있는 것 중에서 자신의 속성을 가지고 있고 다른것과 식별 가능한 것을 말합니다.
#  0   datetime    10886 non-null  object 했을때 object 나오면 string문자열으로 생각하면 편하다? 여기에 한해서 -> datetime으로 바꿔서 나중에 다시써야한다.
#print(type(train))         #<class 'pandas.core.frame.DataFrame'> 나중에 pandas를 numpy로 바꿔야한다.
#print(train.describe())    #train을 자세히 볼 수 있다. count 행의 개수 총 개수  maen 평균값.  std 표준편차 분포도? min 최소값 max최대값 25% 50% 75%의 평균값들 
#print(train.columns) Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp','atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],dtype='object')
#print(train.head()) 위에서부터 5개(default값) 3쓰면 3개
#print(train.tail()) 밑에서부터 5개(default값) 7쓰면 7개~

# train을 분리시켜서 x,y로 나눠야함 x 컬럼8개 y칼럼 1개  datetime casual registered는 빼버림


x = train.drop(['datetime','casual','registered','count'], axis=1)  # axis=1해줘야 열이 삭제됨 axis=0해주면 행삭제(default값)
test_file = test_file.drop(['datetime'], axis=1)

#print(x.columns)    # x의 컬럼이 줄어들고 원하는 8개의값만 나온걸 확인할 수 있다.
#print(x.shape)      # (10886, 8)
y = train['count']  # train의 count만 가져오겠다.
#print(y.shape)            # (10886, )

# 로그변환
y = np.log1p(y) # y모든 값에 1더해주고 log화 시킨다. log1p


#plt.plot(y)
#plt.show()      # 데이터가 넓거나 한쪽으로 치우쳐진 경우에 데이터를 로그변환 해준다. 0값이 있으면 안된다. 모든 값에 1더해놓고 로그화 시킨다.


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=66)  

#2. 모델링      이 모델은 y의 값이 천차만별인 회귀형 모델이다
model = Sequential()
model.add(Dense(100, input_dim=8))    
model.add(Dense(95))
model.add(Dense(90))
model.add(Dense(85))
model.add(Dense(80))
model.add(Dense(75))
model.add(Dense(70))
model.add(Dense(65))
model.add(Dense(60))
model.add(Dense(55))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))   
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 이 모델은 RMSLE로 해아한다고 하네~

#es = EarlyStopping
#es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=30,batch_size=1, verbose=1,validation_split=0.11111111, ) #callbacks=[es]

#4. 평가
loss = model.evaluate(x_test,y_test)   
print('loss값 : ', loss) 

#5. 예측     비교해보면 정확도를 검증할수 있다
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("R2 : ", r2)

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

# y에 log 안씌운값.
# loss값 :  23817.400390625          24274.953125
# R2 :  0.2464691949711344           0.24030994263465288
# RMSE :  154.32886793879092         155.80420626358412

#y값 log화 시킨후   실질적으로 여기서 나오는  rmse값은 이미 log화 시킨 y값을 또 RMSE함수 들어가서 RMSLE값과 비슷하다.
# y에 log 씌운값.
# loss값 :   1.4510016441345215      1.432651162147522          1.4329811334609985          1.964599370956421
#     R2 :   0.25952479581392796     0.26834010256942986        0.2681715744937415          -0.003327715201195458
#  RMSE :    1.2045752284981979      1.1969340635666597         1.1970719045094265          1.4016416861905503


############################# 제출용 제작 ####################################
results = model.predict(test_file)

submit_file['count'] = results  #test_file에서 예측한 값을 results에 담아서 그 값을 submit_file의 count열에 넣어준다.

#print(submit_file[:10]) 확인해보면 count 값이 0에서 model.predict(test_file)에서 나온 결과값이 다 들어가져있는걸 확인 할 수 있다.

submit_file.to_csv(path + '4thtest.csv', index=False) # index를 넣지않겠다는 설정    
