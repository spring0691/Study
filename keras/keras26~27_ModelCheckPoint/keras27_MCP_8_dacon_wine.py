from tensorflow.keras.models import Sequential, Model, load_model      
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies
import time

#1. 데이터 로드 및 정제
path = "../_data/dacon/wine/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sample_Submission.csv')     

x = train.drop(['id','quality'], axis=1)    

Le = LabelEncoder()     

label = x['type']  
Le.fit(label)       
x['type'] = Le.transform(label)    


test_file = test_file.drop(['id'], axis=1)
label2 = test_file['type']
Le.fit(label2)
test_file['type'] =Le.transform(label2) 

y = train['quality']    
y = get_dummies(y)




x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=66)  


scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
    
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    
test_file = scaler.transform(test_file)


#2. 모델링

input1 = Input(shape=(12,))
dense1 = Dense(100, activation='relu')(input1) #
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2) # 
dense4 = Dense(75, activation='relu')(dense3) # 
dense5 = Dense(50, activation='relu')(dense4) # 
dense6 = Dense(10, activation='relu')(dense5)
output1 = Dense(5,activation='softmax')(dense6)
model = Model(inputs=input1,outputs=output1)
      

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_8_wine{krtime}_MCP.hdf5')
model.fit(x_train,y_train,epochs=5000,batch_size=5, verbose=1,validation_split=0.1111111111,callbacks=[es,mcp])

model.save(f"./_save/keras26_8_save_wine{krtime}.h5")

#4. 평가
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



print("======================= 2. load_model 출력 ======================")
model2 = load_model(f"./_save/keras26_8_save_wine{krtime}.h5")
loss2 = model2.evaluate(x_test,y_test)
print('loss2 : ', loss2[0])
print('accuracy2 : ', loss2[1])



print("====================== 3. mcp 출력 ============================")
model3 = load_model(f'./_ModelCheckPoint/keras26_8_wine{krtime}_MCP.hdf5')
loss3 = model3.evaluate(x_test,y_test)
print('loss3 : ', loss3[0])
print('accuracy3 : ', loss3[1])



############################# 제출용 제작 ####################################
results = model.predict(test_file)

results_int = np.argmax(results, axis=1).reshape(-1,1) + 4 

submit_file['quality'] = results_int

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
