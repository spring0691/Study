from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
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
dense1 = Dense(16)(input1) #,activation="relu"
dense2 = Dense(24)(dense1)
dense3 = Dense(32)(dense2) #,activation="relu" 
dense4 = Dense(16)(dense3)
dense5 = Dense(8)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1,outputs=output1)
      

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_7_bike{krtime}_MCP.hdf5')

model.fit(x_train,y_train,epochs=5000,batch_size=50, verbose=1,validation_split=0.11111111,callbacks=[es,mcp])

model.save(f"./_save/keras26_7_save_bike{krtime}.h5")

#4. 평가
print("======================= 1. 기본 출력 =========================")

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

print("======================= 2. load_model 출력 ======================")
model2 = load_model(f"./_save/keras26_7_save_bike{krtime}.h5")
loss2 = model2.evaluate(x_test,y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)

r2 = r2_score(y_test,y_predict2) 
print('r2스코어 : ', r2)

print("====================== 3. mcp 출력 ============================")
model3 = load_model(f'./_ModelCheckPoint/keras26_7_bike{krtime}_MCP.hdf5')
loss3 = model3.evaluate(x_test,y_test)
print('loss3 : ', loss3)

y_predict3 = model3.predict(x_test)

r2 = r2_score(y_test,y_predict3) 
print('r2스코어 : ', r2)


############################# 제출용 제작 ####################################
# results = model.predict(test_file)

# submit_file['count'] = results  
# submit_file.to_csv(path + 'nolog_MaxAbs.csv', index=False)  

'''                                 y값 로그O                                                   y값 로그X
결과정리                  일반레이어                      relu                    일반레이어                         relu
                                                                                    정확도고 자시고 -값때문에 다 리젝당함
안하고 한 결과 
loss값 :            1.4784865379333496          1.3939474821090698              23113.783203125             20336.421875
R2 :                0.2917459249344957          0.3322434796939797              0.2761459118363454          0.36312456133706306  
RMSE :              1.2159303183182582          1.1806555557714267              152.03217668216305          142.60581924576627

MinMax
loss값 :            1.4826191663742065          1.376928687095642               23086.4921875               19986.966796875
R2 :                0.2897662439632225          0.3403961674900349              0.27700049700490914         0.3740685146224785
RMSE :              1.217628490239822           1.1734260670034284              151.94240526589914          141.3752534752431

Standard
loss값 :            1.485615611076355           1.3592065572738647              23078.669921875             20114.212890625  
R2 :                0.2883307752457015          0.3488858125246118              0.27724548877172295         0.3700835382780253
RMSE :              1.2188583566891529          1.1658501305517504              151.91665988700473          141.82457038587344

Robust
loss값 :            1.47902512550354            1.3599188327789307              23229.5859375               20291.291015625
R2 :                0.29148796673764255         0.34854455230668224             0.2725191998441743          0.3645378718555361
RMSE :              1.2161517294252668          1.1661556116484861              152.4125627391536           142.44750081269214

MaxAbs
loss값 :            1.482558012008667           1.36235773563385                23098.552734375             20105.78125
R2 :                0.2897954998115978          0.34737620891107013             0.2766229094144902          0.3703474798313583
RMSE :              1.2176034117913292          1.167200855711969               151.98207616318712          141.7948542871914
'''
