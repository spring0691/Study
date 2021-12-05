from tensorflow.keras.models import Sequential, Model         
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies


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


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)  

scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
    
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    
test_file = scaler.transform(test_file)

print(x_train)


'''
#2. 모델링

input1 = Input(shape=(12,))
dense1 = Dense(20, activation='relu')(input1) #
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(60, activation='sigmoid')(dense2) # 
dense4 = Dense(40)(dense3) # 
dense5 = Dense(20)(dense4) # 
dense6 = Dense(10)(dense5)
output1 = Dense(5,activation='softmax')(dense6)
model = Model(inputs=input1,outputs=output1)
      
# model = Sequential()
# model.add(Dense(100, input_dim=12))    
# model.add(Dense(80, activation='relu')) # 
# model.add(Dense(60)) #
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu')) 
# model.add(Dense(20))
# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(5))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

es = EarlyStopping
es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)

model.fit(x_train,y_train,epochs=5000,batch_size=5, verbose=1,validation_split=0.25,callbacks=[es])

#4. 평가
loss = model.evaluate(x_test,y_test)   
print('loss값 accuracy값 : ', loss) 


############################# 제출용 제작 ####################################
results = model.predict(test_file)

results_int = np.argmax(results, axis=1).reshape(-1,1) + 4

submit_file['quality'] = results_int

acc= str(round(loss[1], 4)).replace(".", "_")
submit_file.to_csv(path+f"result/accuracy_{acc}.csv", index = False)
'''