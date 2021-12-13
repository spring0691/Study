from tensorflow.keras.models import Sequential, Model, load_model      
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D,Input
import numpy as np
import pandas as pd
from pandas import get_dummies
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# ***이번 심장질환 예측대회의 평가지표는 F1 score입니다.*** <-- 이게 무엇인지 공부하세요.



#1. 데이터 로드 및 정제

###자료 3개 로드
path = "/_data/dacon/heart_disease/"   

train = pd.read_csv(path + 'train.csv') 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sample_Submission.csv')     


#print(train) id는 불필요한 인덱스 값이므로 drop시킨다. 여러개의 칼럼들이 수치를 나타내는게 아니라 그 안에 각각 성별 등등의 값을 담고 있다. 
#전부 원핫인코딩해줘야 한다고 생각함. 한거 vs 안한거 수치 비교해서 더 좋은거 쓰기. 
# 또 전에 배운 corr()과 seaborn의 heatmap을 사용해서 연관관계가 낮은 칼럼은 한두개 정도 빼서 비교해보자.
# target이 결과값이므로 y로 빼서 따로 받아준다.
#print(test_file)    # target칼럼이 없다. train에서 fit해서 만들어낸 모델에 test_file을 predict하여 submit_file에 결과를 담아준다.
#print(submit_file)  # target값이 -1인채로 출력되어있다.


x = train.drop(['id','target'], axis=1).drop(index=131,axis=131)         # (151, 13)    -> (150, 13)
test_file = test_file.drop(['id'], axis=1)      # (152, 13) -> submit_file과 행의 개수 동일.
y = train['target'].drop(index=131,axis=131)                           # (151,)  -> (150,)  
#print(x.shape, test_file.shape, y.shape, submit_file.shape)    항상 해보고 확인.


# dacon에 나와있는 정보를 바탕으로 칼럼값들 분석.
# sex -> 0,1이 남녀.   cp -> 1,2,3,4 통증의 종류.     restecg -> 0,1,2 심전도 결과.       exang -> 0,1 없음과 있음.
# slope -> 1,2,3 상승 평탄 하강     thal -> 3,6,7 정상 고정결함 가역결함        target -> 0,1 50%미만과 50%초과로 구분해야함.

#print(np.unique(y, return_counts = True))   # 0,1이 각각 68,83개 있는 이진분류해야할 데이터입니다.

### 원핫인코딩 시작. cp,slope,thal이 0부터 시작하지 않고 연속된 값이 아니므로 pandas 사용해보자.

# one_hot = train['sex','cp','restecg','exang','slope','thal','target']
# one_hot = get_dummies(one_hot)
# print(one_hot)

# x['cp'] = get_dummies(x['cp'])
# print(x.shape)

# one_sex = train['sex']
# one_sex = get_dummies(one_sex)
# #print(one_sex)

# one_restecg = train['restecg']
# one_restecg = get_dummies(one_restecg)
# #print(one_restecg)

# one_exang = train['exang']
# one_exang = get_dummies(one_exang)
# #print(one_exang)

# one_slope = train['slope']
# one_slope = get_dummies(one_slope)
# #print(one_slope)

# one_thal = train['thal']
# one_thal = get_dummies(one_thal)
#print(one_thal)
#우연히 원핫인코딩하다가 얻어 걸린건데 결측치가 나왔다. 3가지의 값만 있어야하는데 0인 값이 나왔다. 이 행을 삭제하자.
#x = train.drop(index=131)  #더 디테일하고 정확한 명령어가 있겠지만 아직 난 초보자이니까 이렇게하자.
#print(x)   행이 1개 삭제된걸 확인 할 수 있다.

y = get_dummies(y)
#print(y)
#제대로 원핫인코딩 되어있나 확인.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)   
#y_train = get_dummies(y_train)
#y_test = get_dummies(y_test) 

### 131번째 행 삭제 + 내가 원하는 7개의 칼럼을 다 원핫인코딩 해야함.
### 근데 일단 너무 완벽하게 다 짜서 해보려고하지말고 적당히 틀을맞춰서 한번 제출해 보고 그 다음에 조금씩 살을 덧붙여보자.



#2. 모델링

model = Sequential()
model.add(Dense(40, input_dim = 13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer= 'adam')

es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=1,validation_split=0.111111, callbacks=[es])#,mcp

#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print(loss)

y_predict = model.predict(x_test)

#y_test = np.argmax(y_test, axis=1)

y_test = y_test.to_numpy()
#y_predict = y_predict.to_numpy()

y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict, axis=1)

print(y_predict)
f1 = f1_score(y_test,y_predict)

print(f1)
#print('f1',f1)
#y_predict_int = #.reshape(-1,1)
#y_test = np.argmax(y_test, axis=1).reshape(-1,1)
#F1 = f1_score(y_test,y_predict_int)

#print('loss : ', loss[0])
#print('accuracy : ', loss[1])

#print('f1 : ', F1)

### 제출

# results = model.predict(test_file)

# results_int = np.argmax(results,axis=1)

# submit_file['target'] = results_int

# acc= str(round(loss[1], 4)).replace(".", "_")
# submit_file.to_csv(path+f"result/accuracy_{acc}.csv", index = False)