from tensorflow.keras.models import Sequential, Model, load_model      
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D,Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import optimizers


#1. 데이터 로드 및 정제
path = "../_data/dacon/wine/"   

train = pd.read_csv(path + 'train.csv') #train.csv를 train에 담아 사용해서 원본은 그대로있다. 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sample_Submission.csv')     

x = train.drop(['id','quality'], axis=1)   # shape(3231, 12)
test_file = test_file.drop(['id'], axis=1) # shape(3231, 12)
y = train['quality']                       # shape(3231,)
#print(y.value_counts())     # 6 5 7 8 4개의 value가 각각 1418 1069 539 108 97순으로 있는 다중분류모델
 
# Le = LabelEncoder() + label = x.type + Le.fit(label) + x.type = Le.transform(label)
x.type = LabelEncoder().fit_transform(x.type)                   # type의 white와 red값이 0,1로 바뀌어있음
test_file.type = LabelEncoder().fit_transform(test_file.type)   # 상동

x['quality'] = y         # x의 데이터셋에 y값을 price라는 이름의 칼럼으로 추가한다. 

#print(x)              #pandas형 데이터라 index와 colmuns의 이름이 나옴. quality열이 추가되어 있는 것 확인.

#print(x.corr())      # price와 어떤 열이 제일 상관관계가 적은지 확인.   .corr() -> 컬럼들의 상관관계를 수치로 보여주는 함수

#########################################################
# import matplotlib.pyplot as plt
# import seaborn as sns   # 조금 더 이쁘게 만들게 도와줌.
# plt.figure(figsize=(10,10))
# sns.heatmap(data=x.corr(), square=True, annot=True, cbar=True)
# # seaborn heatmap 개념정리
# plt.show()
###########################################################
# sulphates 0.027 pH 0.036 total sulfur dioxide 0.044 free sulfur dioxide 0.068 residual sugar 0.045 
# citric acid 0.067 fixed acidity 0.082

#x = x.drop(['pH','sulphates','total sulfur dioxide','free sulfur dioxide','residual sugar','citric acid','fixed acidity','quality'], axis=1)
#test_file = test_file.drop(['pH','sulphates','total sulfur dioxide','free sulfur dioxide','residual sugar','citric acid','fixed acidity'],axis=1)
x = x.drop(['quality'], axis=1) 

#print(x.shape,test_file.shape) #drop 잘되었나 확인.
y = get_dummies(y)  # 원한인코딩 해준후 확인.(3231,5)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)  

scaler =RobustScaler()   #MinMaxScaler()MaxAbsScaler()StandardScaler()

# cnn방식 scaler    
# x_train = scaler.fit_transform(x_train).reshape(len(x_train),5,2,1)
# x_test = scaler.transform(x_test).reshape(len(x_test),5,2,1)  
# test_file = scaler.transform(test_file).reshape(len(test_file),5,2,1)

# dnn방식 scaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  
test_file = scaler.transform(test_file)


#2. 모델링

# Sequtial모델링 이게 더편하긴함
model = Sequential()
model.add(Dense(40, input_dim=12))    
model.add(Dense(60, activation='relu')) # 
model.add(Dropout(0.5))
model.add(Dense(80)) #
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


# Model함수 써서 Input 사용해서 모델링
# input1 = Input(shape=(5,2,1))
# conv1  = Conv2D(4,kernel_size=(2,1),strides=1,padding='valid',activation='relu')(input1) # 4,2,4
# maxf   = MaxPooling2D(2,2)(conv1)                                                       # 2,1,4
# conv2  = Conv2D(4,kernel_size=(2,1),strides=1,padding='valid',activation='relu')(conv1) # 1,1,4
# fla    = Flatten()(conv2)
# dense1 = Dense(16,activation="relu")(fla) #
# dense2 = Dense(24)(dense1)
# drop1  = Dropout(0.5)(dense2)
# dense3 = Dense(10,activation="relu")(drop1) # 
# output1 = Dense(5,activation='softmax')(dense3)
# model = Model(inputs=input1,outputs=output1)
      

#3. 컴파일, 훈련
#sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # SGD

es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience = 25, verbose = 1)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_8_wine{krtime}_MCP.hdf5')
model.fit(x_train,y_train,epochs=5000,batch_size=5, verbose=1,validation_split=0.1111111111,callbacks=[es])#,mcp


#4. 평가
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

acc = str(round(loss[1],4))
model.save(f"./_save/keras32_8_wine{acc}.h5")


############################# 제출용 제작 ####################################
results = model.predict(test_file)

results_int = np.argmax(results, axis=1).reshape(-1,1) + 4 

# submit_file['quality'] = results

# submit_file.to_csv(path+f"result/accuracy_{acc}.csv", index = False)


'''                                                                              
결과정리                  일반레이어                      relu                   
                                                                                    
안하고 한 결과 
loss값 :                                            1.0440469980239868
accuracy :                                          0.5123456716537476

MinMax
loss값 :                                            0.9981087446212769
accuracy :                                          0.5432098507881165

Standard
loss값 :                                            0.9914785027503967
accuracy :                                          0.540123462677002

Robust
loss값 :                                            1.031555414199829       1.0050442218780518
accuracy :                                          0.5617284178733826      0.5771604776382446

MaxAbs
loss값 :                                            1.0235952138900757
accuracy :                                          0.5246913433074951
''' 
