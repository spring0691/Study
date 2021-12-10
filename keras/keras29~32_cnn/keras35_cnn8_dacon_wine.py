from tensorflow.keras.models import Sequential, Model, load_model      
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D,Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies
import time

from tensorflow.python.keras.regularizers import get

#1. 데이터 로드 및 정제
path = "../_data/dacon/wine/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sample_Submission.csv')     

x = train.drop(['id','quality'], axis=1)   
test_file = test_file.drop(['id'], axis=1) 
y = train['quality']    


#print(x.shape)  #(3231, 12)
#print(y.shape)  #(3231, 5)


Le = LabelEncoder()     

label = x['type']  
Le.fit(label)       
x['type'] = Le.transform(label)    


label2 = test_file['type']
Le.fit(label2)
test_file['type'] =Le.transform(label2) 


# numpy pandas로 변환후 pandas의 제공기능인 index정보와 columns정보를 확인할수있다.
# 원래 판다스라서 그냥한다.

#print(x)               #index와 colmuns의 이름이 나옴.

#print(x.corr())        # 칼럼들의 서로서로의 상관관게를 수치로 확인할 수 있다.    절대값클수록 양 or 음의 상관관계 0에 가까울수록 서로 영향 없음

x['y~~'] = y         # xx의 데이터셋에 y값을 price라는 이름의 칼럼으로 추가한다. 원본데이터는 그대로있다.    열 추가하는 방법.

#print(x)              # price열이 추가되어 있는 것 확인.

# print(x.corr())      # price와 어떤 열이 제일 상관관계가 적은지 확인.

#########################################################
# import matplotlib.pyplot as plt
# import seaborn as sns   # 조금 더 이쁘게 만들게 도와줌.
# plt.figure(figsize=(10,10))
# sns.heatmap(data=x.corr(), square=True, annot=True, cbar=True)
# # seaborn heatmap 개념정리
# plt.show()
###########################################################
x = x.drop(['pH','sulphates','y~~'], axis=1)  
test_file = test_file.drop(['pH','sulphates'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)  


scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()
    

x_train = scaler.fit_transform(x_train).reshape(len(x_train),5,2,1)
x_test = scaler.transform(x_test).reshape(len(x_test),5,2,1)  
test_file = scaler.transform(test_file).reshape(len(test_file),5,2,1)

y_train = get_dummies(y_train)
y_test = get_dummies(y_test)
#2. 모델링

input1 = Input(shape=(5,2,1))
conv1  = Conv2D(4,kernel_size=(2,1),strides=1,padding='valid',activation='relu')(input1) # 4,2,4
maxf   = MaxPooling2D(2,2)(conv1)                                                       # 2,1,4
conv2  = Conv2D(4,kernel_size=(2,1),strides=1,padding='valid',activation='relu')(conv1) # 1,1,4
fla    = Flatten()(conv2)
dense1 = Dense(16,activation="relu")(fla) #
dense2 = Dense(24)(dense1)
drop1  = Dropout(0.5)(dense2)
dense3 = Dense(10,activation="relu")(drop1) # 
output1 = Dense(5,activation='softmax')(dense3)
model = Model(inputs=input1,outputs=output1)
      

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_8_wine{krtime}_MCP.hdf5')
model.fit(x_train,y_train,epochs=5000,batch_size=5, verbose=1,validation_split=0.1111111111,callbacks=[es])#,mcp

#model.save(f"./_save/keras26_8_save_wine{krtime}.h5")

#4. 평가
# print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



############################# 제출용 제작 ####################################
results = model.predict(test_file)

#results_int = np.argmax(results, axis=1).reshape(-1,1) + 4 

submit_file['quality'] = results

acc= str(round(loss[1], 4)).replace(".", "_")
submit_file.to_csv(path+f"result/accuracy_{acc}.csv", index = False)


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
