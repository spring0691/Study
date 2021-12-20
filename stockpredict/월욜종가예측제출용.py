from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Activation,Dropout, Input,concatenate,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd , numpy as np
from sklearn.metrics import r2_score 

def split_xy5(dataset, time_steps, y_column):                
    x,y = list(), list()                                    

    for i in range(len(dataset)):                           
        x_end_number = i + time_steps                       
        y_end_number = x_end_number + y_column             

        if y_end_number > len(dataset):                        
            break

        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x),np.array(y)


path = "./"   

쌈쏭 = pd.read_csv(path + '삼성전자.csv', encoding='ANSI',index_col=0, header=0,thousands = ',' ).iloc[:239,:].sort_values(['일자'],ascending=[True])  # 2021년 데이터만 사용   
끼윰 = pd.read_csv(path + '키움증권.csv', encoding='ANSI',index_col=0, header=0,thousands = ',' ).iloc[:239,:].sort_values(['일자'],ascending=[True]) 

s = 쌈쏭[['시가','고가','저가','종가']].values            
k = 끼윰[['시가','고가','저가','종가']].values         

d = 6
sam1220 = s[-d:]            # reshape해주기전 12일~17일의 시가 고가 저가 종가 값이 담긴거 확인.
kium1220 = k[-d:]           # reshape해주기전 12일~17일의 시가 고가 저가 종가 값이 담긴거 확인.
x1,y1 = split_xy5(s,d,1)    # 마지막값은  12/09~토일지나서12/16까지 들어가고 y는 그 다음날의 종가 들어간거 확인.
x2,y2 = split_xy5(k,d,1)    # 키움역시 같음을 확인.    
# 검증완료. 이로써 sam1220과kium1220을 넣으면 정말로 12월20일의 종가를 predict해볼수 있음을 확인했다.
  
### shape변환 단계. 

# DNN 사용시    
# x1 = x1.reshape(len(x1),-1)
# x2 = x2.reshape(len(x2),-1)
# sam1220 = sam1220.reshape(1,-1)
# kium1220 = kium1220.reshape(1,-1)

# RNN 사용시       
sam1220 = sam1220.reshape(1,sam1220.shape[0],sam1220.shape[1])
kium1220 = kium1220.reshape(1,kium1220.shape[0],kium1220.shape[1])

x1_train,x1_test,y1_train,y1_test,x2_train,x2_test,y2_train,y2_test = train_test_split(x1,y1,x2,y2, train_size=0.75, shuffle=True, random_state=66)  #

model = load_model("./1220_LSTM_d6_삼성0.95&[[77222.09]],키움0.96&[[110002.45]].h5")


### 평가 예측.
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test], batch_size=1)  

print("----------------------loss값-------------------------")
print('loss : ',loss)

print("=====================r2score=========================")
y_predict = model.predict([x1_test,x2_test])    

r2_sam = np.array(round(r2_score(y1_test,y_predict[0]),2))
r2_kium = np.array(round(r2_score(y2_test,y_predict[1]),2))

print('삼성r2_score : ', r2_sam)
print('키움r2_score : ', r2_kium)

print("=====================12/20예측종가=========================")
주식1220 = np.array(model.predict([sam1220,kium1220]))
삼성 = 주식1220[0]
키움 = 주식1220[1]
print('12/20삼전 예측종가 : ',삼성)
print('12/20키움 예측종가 : ',키움)

