from numpy.lib.function_base import vectorize
from sklearn.datasets import fetch_covtype              # 데이터 입력받음.
from tensorflow.keras.models import Sequential          # 이 아래는 말안해도 다 알지?
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical       # onehotencoding 이건 명칭이고 to_categorical함수 이용해서 변환하는게 실제 하는거.

#1. 데이터 
datasets = fetch_covtype()
#print(datasets.DESCR)           # 데이터셋 및 컬럼에 대한 설명 
#print(datasets.feature_names)   # 컬럼,열의 이름들                                                      

x = datasets.data   # 싸이킷런에서 이런식으로 제공해서 이렇게 분리하는거다. 붙여서 제공해줬기 때문에
y = datasets.target

#print(x.shape, y.shape)     # (581012,54) (581012,) 행과 열의 개수를 확인할수 있다
#print(np.unique(y))         # [1 2 3 4 5 6 7] 7개인거확인 이게 뭘 의미하냐 이 모델은 다중분류형태로 모델링 해줘야 한다~

# one hot encoding 3가지 방법이 있다.
y = to_categorical(y)   # onehotencoding해줘서 배열형태로 변환후 몇개의 값만 확인해보자.    해주는 이유는? 모든 값들을 동일값 1로 통일시켜서 fit 학습시키려고
#print(y.shape)  # (581012, 8) 칼럼이8개이니까 softmax에 8을 넣는다.
#print(y[:10]) #[0. 0. 0. 0. 0. 1. 0. 0.],[0. 0. 0. 0. 0. 1. 0. 0.],[0. 0. 1. 0. 0. 0. 0. 0.] 각각의 y값이 배열(?) 형태로 변환된것을 확인할 수 있다.
# 배열형태로 변환후 unique(y)처럼 중복값없이 8가지 배열 형태 다 못보나? <---------------------------------------------- 질문할거
# # [10000000][01000000][001000000][00010000][00001000][00000100][00000010][00000001] 8가지 값에 0~7 or 1~8(이렇게도 할수있나?)이 각각 들어가야하는데 한 값이 빠져있다.
# 자리로는 [1,0,0,0,0,0,0,0] 8자리를 차지하는데 유니크값은 7개이다 
# 이게 무엇을 의마하냐 내 추측으로는 값은 0~7까지 또는 1~8까지 8개인데 1개의 값을안 쓰고 7개만 사용했다. 그래서 실제로 찍어보면 [0,0,0,0,1,0,0,0] 처럼 8자리가 나온다.
# 그럼 일단 softmax에는 8개를 넣는다. 
# 이건 이제 일종의 코딩 어렵게하려는 함정? 같은 개념이 아닐까


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)    # 여기는 뭐 모르면 공부 접어야지

# 숙제하기 위해 print(len(x_train))해서 길이를 구해보고 가자
#z = [1,2,3,4,5]  print(len(z)) 혹시 모르니까 5개는 5개로 출력 되는거 확인.
#print(len(x_train))    464809 개 인걸 확인할 수 있다.



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
model.add(Dense(8, activation='softmax'))       #(0.a, 0.b, 0.c, 0.d, 0.e, 0.f, 0.g, 0.h) a+...+h = 1
#회귀모델 activation = linear (default값) 이진분류 sigmoid 다중분류 softmax
#  
# 행이 540000개이길래 500000만개부터 5만개씩줄여서 50만 40만 30만 20만 10만 이렇게 하려고했는데  cpu가 절대못한다고 뭐라고 함.;;

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)

hist = model.fit(x_train,y_train,epochs=1000000,batch_size=1000, verbose=1,validation_split=0.2, callbacks=[es])    

#batch_size 통째로 빼 보고 해보기 디폴트 사이즈 몇인지 알아보기  로딩되는 과정에서 1epcoh에 값이 몇인지 확인해보기. 데이터 수 / batch사이즈 값만큼 반복한다. 나머지도 1번으로 계산한다.
# 464809 * 0.8이 train 데이터고 나머지 0.2가 validation 데이터인데 train데이터만 fit에 들어가기때문에 train데이터의 개수를 구해야한다.
# 계산기로 464809 * 0.8 해보면 371,847.2 -> 371847 or 371848 둘중 하나 
# 실행시켜보면 37~~~~ 값만큼 반복하지는 않고 11621만큼 반복한다 이 값을 나눠보면 bacth_size의 default값을 구할 수 있다 
# 아마 train데이터 개수가 딱 맞아떨어지지 않아서 11621 반복값은 나머지 연산처리되서 + 1이 들어간 값일 것이다. 따라서 11620으로 나눠줘야한다.
# case1 371847 / 11620 = 32.00060240963855
# case2 371848 / 11620 = 32.00068846815835
# train데이터의 개수가 1개정도 차이난다고 쳐도 batch사이즈 기본값은 32이고 거기에 나머지가 좀 남은거 1번 연산 더해서 11621이 나온거 같다.
# batch_size: 정수 혹은 None. 경사 업데이트 별 샘플의 수. 따로 정하지 않으면 batch_size는 디폴트 값인 32가 된다. 
# 구글링 해봤는데 내 계산이 맞았다. 확인 끝났으면 batch_size=10000정도로 줘서 일단 해본다.

#print(len(x_train)) #model.fit에서 다시 train과 validation으로 나눠주니까 여기서 측정하면 나눠진후의 x_train값이 나올줄 알았는데 464809가 나왔다.
# model.fit안에서 자체적으로 나눠서 계산해주고 그 밖까지 값이 저장되지는 않는거같다.

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)   
print('loss : ', loss[0])          # batch_size=1000 loss :  0.6324978470802307         batch_size=100  loss :  0.639609694480896           loss :  0.637958824634552
print('accuracy : ', loss[1])      # batch_size=1000 accuracy :  0.725635290145874      batch_size=100   accuracy :  0.7248436212539673     accuracy :  0.7228556871414185

results = model.predict(x_test[:15])   
#predict는 metrics=['accuracy'] 못쓰나? 일일히 하나하나 다 비교해봐야하나? 제일 마지막단계가 predict이긴한데 그래도 어떻게 안되나? 
#predict() got an unexpected keyword argument 'metrics' 처음부터 기능이 없는거같다. 예측하고 이제 자기가 모델링한거 믿고 결과값 그냥 쓰는거구나.
print(y_test[:15])
print(results)

