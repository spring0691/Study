# 시작!
# 20~30년전의 보스턴 집값 데이터를 활용하는 예제.
# 오늘 배운 모든 것을 총동원해서 알고리즘 완성해보기 
# train_test set 0.6~0.8 사이 , r2 score 0.8이상 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target
'''
print(x)    # x내용물 확인
print(y)    # y내용물 확인
print(x.shape) # x형태
print(y.shape) # y형태
print(datasets.feature_names) # 컬럼,열의 이름들
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
'''
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8125, shuffle=True, random_state=66)

#2. 모델링 
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(120))
model.add(Dense(160))
model.add(Dense(140))
model.add(Dense(120))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train,y_train,epochs=200, batch_size=1,validation_split=0.23)

#4. 평가 , 예측
#loss = model.evaluate(x_test,y_test)
#print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)

# 왜 값이 더 떨어지지..??
# r2스코어 :  0.7579988043990163
# r2스코어 :  0.7846635062486282 