from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))    

#1. 데이터 로드 및 정제
path = "../_data/kaggle/bike/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     

x = train.drop(['datetime','casual','registered','count'], axis=1)  
test_file = test_file.drop(['datetime'], axis=1)

y = train['count']  

y = np.log1p(y) 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)  

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
    
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    
test_file = scaler.transform(test_file)


#2. 모델링

input1 = Input(shape=(8,))
dense1 = Dense(16,activation="relu")(input1) #
dense2 = Dense(24)(dense1)
drop1  = Dropout(0.2)(dense2)
dense3 = Dense(32,activation="relu")(drop1) # 
dense4 = Dense(16)(dense3)
drop2  = Dropout(0.4)(dense4)
dense5 = Dense(8)(drop2)
output1 = Dense(1)(dense5)
model = Model(inputs=input1,outputs=output1)
      

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_7_bike{krtime}_MCP.hdf5')

model.fit(x_train,y_train,epochs=5000,batch_size=50, verbose=1,validation_split=0.11111111,callbacks=[es])#,mcp

#model.save(f"./_save/keras26_7_save_bike{krtime}.h5")

#4. 평가
print("======================= 1. 기본 출력 =========================")

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

# print("======================= 2. load_model 출력 ======================")
# model2 = load_model(f"./_save/keras26_7_save_bike{krtime}.h5")
# loss2 = model2.evaluate(x_test,y_test)
# print('loss2 : ', loss2)

# y_predict2 = model2.predict(x_test)

# r2 = r2_score(y_test,y_predict2) 
# print('r2스코어 : ', r2)

# rmse = RMSE(y_test,y_predict)
# print("RMSE : ", rmse)

# print("====================== 3. mcp 출력 ============================")
# model3 = load_model(f'./_ModelCheckPoint/keras26_7_bike{krtime}_MCP.hdf5')
# loss3 = model3.evaluate(x_test,y_test)
# print('loss3 : ', loss3)

# y_predict3 = model3.predict(x_test)

# r2 = r2_score(y_test,y_predict3) 
# print('r2스코어 : ', r2)


# rmse = RMSE(y_test,y_predict)
# print("RMSE : ", rmse)

############################# 제출용 제작 ####################################
# results = model.predict(test_file)

# submit_file['count'] = results  
# submit_file.to_csv(path + 'nolog_MaxAbs.csv', index=False)  

'''                            y값 로그O (x하면-값때문에 다 리젝당함)                                                  
결과정리                  일반레이어                      relu                  drop+relu          
                                                                                    
안하고 한 결과 
loss값 :            1.4784865379333496          1.3939474821090698           
R2 :                0.2917459249344957          0.3322434796939797              
RMSE :              1.2159303183182582          1.1806555557714267             

MinMax
loss값 :            1.4826191663742065          1.376928687095642              
R2 :                0.2897662439632225          0.3403961674900349              
RMSE :              1.217628490239822           1.1734260670034284              

Standard
loss값 :            1.485615611076355           1.3592065572738647             
R2 :                0.2883307752457015          0.3488858125246118              
RMSE :              1.2188583566891529          1.1658501305517504             

Robust
loss값 :            1.47902512550354            1.3599188327789307              
R2 :                0.29148796673764255         0.34854455230668224           
RMSE :              1.2161517294252668          1.1661556116484861            

MaxAbs
loss값 :            1.482558012008667           1.36235773563385         1.3601329326629639  
R2 :                0.2897954998115978          0.34737620891107013      0.348441930059461     
RMSE :              1.2176034117913292          1.167200855711969        1.166247458909339      
'''
