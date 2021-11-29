from sklearn.datasets import fetch_covtype              # 데이터 입력받음.
from tensorflow.keras.models import Sequential          # 이 아래는 말안해도 다 알지? 모르면 공부 접자 예람아
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
y = to_categorical(y)   # onehotencoding해줘서 배열형태로 변환후 몇개의 값만 확인해보자.    해주는 이유는? 모든 값들을 동일값 1로 통일시켜서 모델이 공정하게(?) fit 학습시키려고
#print(y[:10]) #[0. 0. 0. 0. 0. 1. 0. 0.],[0. 0. 0. 0. 0. 1. 0. 0.],[0. 0. 1. 0. 0. 0. 0. 0.] 각각의 y값이 배열(?) 형태로 변환된것을 확인할 수 있다.
# 자리로는 [1,0,0,0,0,0,0,0] 8자리를 차지하는데 유니크값은 7개이다 
# 이게 무엇을 의마하냐 내 추측으로는 값은 0~7까지 1개인데 0값을안 쓰고 7개만 사용했다. 그래서 실제로 찍어보면 [0,0,0,0,1,0,0,0] 처럼 8자리가 나온다.
# 그럼 일단 softmax에는 8개를 넣는다. 이론상 결과로는 8개의 값이 나올때 제일 앞에 제일 큰 값이 올 수가없다. 왜냐하면 데이터를 그렇게 안 넣었으니까.
# 이건 이제 일종의 코딩 어렵게하려는 함정? 같은 개념이 아닐까... [[[    제가 이해한게 맞나요 선생님   ]]]
#print(y.shape)  # (581012, 8)   # 위의 이유때문에 8이 나온거고 softmax에 8을 넣어주면 된다.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)    # 여기는 뭐 모르면 공부 접어야지

# 숙제하기 위해 print(len(x_train))해서 길이를 구해보고 가자
#z = [1,2,3,4,5]  print(len(z)) 혹시 모르니까 5개는 5개로 출력 되는거 확인.
#print(len(x_train))    464809 개 인걸 확인할 수 있다.

#2. 모델링 모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_dim=54))    
model.add(Dense(45))   
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(8, activation='softmax'))   
#회귀모델 activation = linear (default값) 이진분류 sigmoid 다중분류 softmax 
# 행이 540000개이길래 500000만개부터 5만개씩줄여서 50만 40만 30만 20만 10만 이렇게 하려고했는데  cpu가 절대못한다고 뭐라고 함.;;
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)

hist = model.fit(x_train,y_train,epochs=1000000,batch_size=100, verbose=1,validation_split=0.2, callbacks=[es])    
#batch_size 통째로 빼 보고 해보기 디폴트 사이즈 몇인지 알아보기  로딩되는 과정에서 1epcoh에 값이 몇인지 확인해보기. 데이터 수 / batch사이즈 값만큼 반복한다. 나머지도 1번으로 계산한다.
# 464809 * 0.8이 train 데이터고 나머지 0.2가 validation 데이터인데 train데이터만 fit에 들어가기때문에 train데이터의 개수를 구해야한다.
# 계산기로 464809 * 0.8 해보면 371,847.2 -> 371847 or 371848 둘중 하나 
# 실행시켜보면 37~~~~ 값만큼 반복하지는 않고 11621만큼 반복한다 이 값을 나눠보면 bacth_size의 default값을 구할 수 있다 
# 아마 train데이터 개수가 딱 맞아떨어지지 않아서 11621 반복값은 나머지 연산 + 1이 들어간 값일 것이다. 따라서 11620으로 나눠줘야한다.
# case1 371847 / 11620 = 32.00060240963855
# case2 371848 / 11620 = 32.00068846815835
# train데이터의 개수가 1개정도 차이난다고 쳐도 batch사이즈 기본값은 32이고 거기에 나머지가 좀 남은거 1번 연산 더해서 11621이 나온거 같다.
# batch_size: 정수 혹은 None. 경사 업데이트 별 샘플의 수. 따로 정하지 않으면 batch_size는 디폴트 값인 32가 됩니다. 
# 구글링 해봤는데 내 계산이 맞았다. 확인 끝났으면 batch_size=10000정도로 화끈하게 줘서 얼른해야지 32로하면 밤새겠다.

#print(len(x_train)) #model.fit에서 다시 train과 validation으로 나눠주니까 여기서 측정하면 나눠진후의 x_train값이 나올줄 알았는데 464809가 나왔다.
# model.fit안에서 자체적으로 나눠서 계산해주고 그 밖까지 값이 저장되지는 않는거같다.
#4. 평가, 예측
loss = model.evaluate(x_test,y_test)   
print('loss : ', loss[0])          # batch_size=1000 loss :  0.6324978470802307         batch_size=100  loss :  0.639609694480896
print('accuracy : ', loss[1])      # batch_size=1000 accuracy :  0.725635290145874      batch_size=100   accuracy :  0.7248436212539673

results = model.predict(x_test[:7])
print(x_test[:7])
print(y_test[:7])
print(results)
# y_test랑 results까서 비교해보자.
# [[0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]]
# e의 -값이 제일 적은거만 추려서 비교하면 쉽게 할 수 있다.
# [[3.68047462e-16 7.27554202e-01 2.09844559e-01 7.22368698e-09 ,1.08441413e-12 9.06930596e-04 1.42577008e-08 6.16942756e-02]   [0,1,0,0,0,0,0,0]
#  [1.05762355e-14 8.05222020e-02 9.18675601e-01 5.12027236e-06 ,4.16913863e-05 6.02370070e-04 1.46435763e-04 6.57679857e-06]   [0,0,1,0,0,0,0,0] 
#  [6.34847047e-13 8.24427545e-01 1.65983200e-01 1.39828032e-07 ,1.81838433e-10 8.24665301e-04 8.41224755e-06 8.75606295e-03]   [0,1,0,0,0,0,0,0]
#  [2.17151752e-12 7.83448145e-02 9.15893912e-01 5.12067600e-05 ,2.43551272e-04 5.05265128e-03 3.49442795e-04 6.44215106e-05]   [0,0,1,0,0,0,0,0]
#  [1.20308816e-14 4.63518977e-01 5.19259691e-01 3.68038718e-06 ,2.50423831e-11 4.29757778e-03 1.63658351e-05 1.29037248e-02]   [0,0,1,0,0,0,0,0]
#  [3.50175133e-17 2.77061820e-01 6.92409515e-01 6.92533922e-06 ,4.20396856e-10 3.04279365e-02 8.22865331e-05 1.14363165e-05]   [0,0,1,0,0,0,0,0]
#  [6.85345877e-14 1.77210420e-01 8.21564496e-01 6.52239032e-06 ,4.02984654e-07 6.61129714e-04 2.53149319e-05 5.31783618e-04]]  [0,0,1,0,0,0,0,0]
# evaluate에서는 비록 0.72의 정확도를 보였지만 7개만 뽑아서 predict한 값은 다 맞췄다.

# 집가서 필기 업데이트.