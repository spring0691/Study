#0.내가쓸 기능들 import
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Activation,Dropout, Input,concatenate,LSTM,Conv1D, Bidirectional
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
        tmp_y = dataset[x_end_number:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x),np.array(y)


#1.데이터로드 및 정제

# 로드영역   

path = "./"   

쌈쏭 = pd.read_csv(path + '삼성전자.csv', encoding='ANSI',index_col=0, header=0,thousands = ',' ).iloc[:239,:].sort_values(['일자'],ascending=[True])  # 2021년 데이터만 사용   
끼윰 = pd.read_csv(path + '키움증권.csv', encoding='ANSI',index_col=0, header=0,thousands = ',' ).iloc[:239,:].sort_values(['일자'],ascending=[True]) 

#삼성&키움.columns ['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)','신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

#칼럼분석
'''
시가 - 주식시장이 열렸을 때 바로 그 순간의 주가를 말합니다.
고가 - 그날 장중 가장 높았던 가격.
저가 - 그날 장중 가장 낮았던 가격.
종가 - 주식시장이 닫히는 순간의 주가를 말합니다.
전일비 - 전날과 비교해서 주식의 가격이 올랐는지, 아니면 내렸는지를 표시 
등락률 - 전일비/전날종가*100 전일 종가에 대한 오늘 현재주가의 백분율
거래량 - 주식시장에서 매매된 주식의 수를 말하는데요, 거래량이 1,000주라고 하면 매도(판매) 1,000주, 매수량(구매)이 1,000주라는 의미가 됩니다. 
금액(거래대금) - 거래된 주식의 가격과 거래량을 곱한 금액을 말합니다.
신용비 - 총발행주식수 대비 신용으로(빚내서) 주식을 산 비율.
개인 - 개인투자자를 의미함. 칼럼에서의 값은 개인투자자의 거래량
기관 - 주식에서 기관은 증권회사 같은 기업이 고객의 돈을 투자받아서 거래하는걸 의미함.  칼럼에서의 값은 기관의 거래량
외인 - 외국인개인투자자. 칼럼은 ~의 거래량
외국계 - 외국기관투자. 칼럼은 ~의 거래량
프로그램 - 프로그램을 사용해서 돌리는 거래량.
외인비 - 외국인 투자자의 비율.
'''

#print(쌈쏭.describe()) 일단 좀 훑어봐봄. 뭐가뭔진 잘 몰르겠지만...
#print(쌈쏭.info()) 여기서 이제 같은 값이라도 int64와 float64가 있는걸 알수있다. <-- 이거때문에 오류 많이떴음.

#------------------------------------------------------------------------------------------------------------------------------ 기초데이터세팅 + 행 열 개수같게.

# 문제 - 월: 종가   

s = 쌈쏭[['시가','종가']].values              
k = 끼윰[['시가','종가']].values        
d = 6

#sam1220 = pd.DataFrame(s[-5:]).drop([3]).values
sam1220 = s[-d:]
# [[77200 78300 76500 76800]
#  [76500 77200 76200 77000]
#  [76400 77600 76300 77600]
#  [78500 78500 77400 77800]
#  [76800 78000 76800 78000]]       reshape해주기전 13일~17일의 시가 고가 저가 종가 값이 담긴거 확인.


#kium1220 = pd.DataFrame(k[-5:]).drop([3]).values
kium1220 = k[-d:]
# [[107000 109500 107000 107500]
#  [106500 109000 106500 107000]
#  [107000 108000 106500 107500]
#  [109500 109500 107000 107500]
#  [107000 109500 106500 109500]]   reshape해주기전 13일~17일의 시가 고가 저가 종가 값이 담긴거 확인.

x1,y1 = split_xy5(s,d,1)
# [[[77400 77600 76800 76900]
#   [77200 78300 76500 76800]
#   [76500 77200 76200 77000]
#   [76400 77600 76300 77600]
#   [78500 78500 77400 77800]]]   [[78000]]     분명하게 마지막값은  12/10~토일지나서12/16까지 들어가고 y는 그 다음날의 종가 들어간거 확인.


x2,y2 = split_xy5(k,d,1)
# [[[108000 108500 106500 106500]
#   [107000 109500 107000 107500]
#   [106500 109000 106500 107000]
#   [107000 108000 106500 107500]
#   [109500 109500 107000 107500]]]   [[109500]]    키움역시 같음을 확인.

# 검증완료. 이로써 sam1220과kium1220을 넣으면 정말로 12월20일의 종가를 predict해볼수 있음을 확인했다.


#---------------------------------------------------------------------------------------------------------   

# shape변환 단계. 

# DNN 사용시    2차원으로 변환
# x1 = x1.reshape(len(x1),-1)
# x2 = x2.reshape(len(x2),-1)
# sam1220 = sam1220.reshape(1,-1)
# kium1220 = kium1220.reshape(1,-1)

# RNN 사용시    3차원으로 변환   
sam1220 = sam1220.reshape(1,sam1220.shape[0],sam1220.shape[1])
kium1220 = kium1220.reshape(1,kium1220.shape[0],kium1220.shape[1])

#---------------------------------------------------------------------------------------------------------- 모델링 들어가기전 마지막 작업 각 모델에 맞게 차원변환

# train & test분리    무조건 train & test 하고 scaler.
x1_train,x1_test,y1_train,y1_test,x2_train,x2_test,y2_train,y2_test = train_test_split(x1,y1,x2,y2, train_size=0.75, shuffle=True, random_state=66)  #

#********************* load_model 사용시 #2. #3. 전부 주석 걸어주시면 됩니다.     *******************

#model = load_model("./1220_LSTM_d6_삼성0.95&[[77222.09]],키움0.96&[[110002.45]].h5")

#************************************************************************************************

모델 = 'LSTM'

#2.모델링   각 데이터에 알맞게 튜닝

# DNN Dense->2차원    CNN Conv2d->3차원 Conv1d->2차원     RNN SimpleRNN,LSTM,GRU->2차원

#삼성 x값 인풋

#DNN - 주석해제하여 사용
# input11 = Input(shape=(x1_train.shape[1]))
# dense11 = Dense(64, activation='relu')(input11)
# #dense11 = Conv1D(64,(2), activation='relu')(input11)

#RNN - 주석해제하여 사용
input11 = Input(shape=(x1_train.shape[1],x1_train.shape[2]))
rnn11 = Bidirectional(LSTM(64, activation='relu'))(input11)
dense11 = Dense(48, activation='relu')(rnn11)
#drop1 = Dropout(0.5)(dense11)

dense12 = Dense(32)(dense11)
dense13 = Dense(16, activation='relu')(dense12)
output11 = Dense(8)(dense13)

#키움 x값 인풋
#DNN - 주석해제하여 사용
# input21 = Input(shape=(x2_train.shape[1]))
# dense21 = Dense(64, activation='relu')(input21)
# #dense21 = Conv1D(64,(2), activation='relu')(input21)


#RNN - 주석해제하여 사용
input21 = Input(shape=(x1_train.shape[1],x1_train.shape[2]))
rnn21 = LSTM(64, activation='relu')(input21)   #Bidirectional()
dense21 = Dense(48, activation='relu')(rnn21)
#drop2 = Dropout(0.5)(dense21)

dense22 = Dense(32)(dense21)
dense23 = Dense(16, activation='relu')(dense22)
output21 = Dense(8)(dense23)


merge = concatenate([output11,output21])

#삼성종가
merge11 = Dense(8, activation='relu')(merge)
merge12 = Dense(4)(merge11)
sam_output = Dense(1)(merge12)

#키움종가
merge21 = Dense(8, activation='relu')(merge)
merge22 = Dense(4)(merge21)
kium_output = Dense(1)(merge22)

model = Model(inputs=[input11, input21], outputs=[sam_output, kium_output])


#3.컴파일,훈련


model.compile(loss='mae', optimizer='adam')                                                           
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)  
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs=10000, batch_size=1,validation_split=0.25,verbose=1,callbacks=[es])  


#4.평가,예측        

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

#***************************** model.save ******************************

#model.save(f"./1220_d{d}_{모델}_삼성{r2_sam}&{삼성},키움{r2_kium}&{키움}.h5")

#***********************************************************************
