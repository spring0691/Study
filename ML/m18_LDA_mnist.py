# LDA활용해서 제작해보기.

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np,time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

scaler = MinMaxScaler()   
use_x_train = scaler.fit_transform(x_train)     
use_x_test = scaler.transform(x_test) 

lda_name_dict = {'0.938':7,'0.973':8,'1,0':9}
lda_result_dict = {}

for lda_acc,lda_num in lda_name_dict.items():
      
    lda = LinearDiscriminantAnalysis(n_components=lda_num)  # n_components= 최대 9개.
    x_train = lda.fit_transform(use_x_train,y_train)
    x_test = lda.transform(use_x_test)
    # lda_EVR = lda.explained_variance_ratio_
    # cumsum = np.cumsum(lda_EVR) 
    # [0.2392286  0.44103854 0.61953549 0.72606121 0.82012832 0.88918857 0.93892603 0.9732168  1.        ] 요것도 explained_variance_ratio_가 있다

    #2. 모델링
    input1 = Input(shape=(lda_num,))            
    dense1 = Dense(100)(input1)
    dense6 = Dropout(0.2)(dense1)
    dense2 = Dense(80)(dense6)
    dense3 = Dense(60,activation="relu")(dense2)
    dense7 = Dropout(0.4)(dense3)
    dense4 = Dense(40,activation="relu")(dense7)
    dense5 = Dense(20)(dense4)
    output1 = Dense(10,activation='softmax')(dense5)
    model = Model(inputs=input1,outputs=output1)

    #3. compile 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc']) 
    es = EarlyStopping(monitor="val_acc", patience=50, mode='max',verbose=1,baseline=None, restore_best_weights=True)
    start = time.time()
    model.fit(x_train,y_train,epochs=10000, batch_size=100,validation_split=0.2, callbacks=[es])#,mcp
    end = time.time()
    #4. 평가 예측

    loss = model.evaluate(x_test,y_test)
    timee = np.round(end - start,4)
    acc = np.round(loss[1],4)

    lda_result_dict[f'{lda_acc}_time&acc']=[timee,acc]
    
for key, value in lda_result_dict.items():
        print(key, value)
        
'''                     DNN 기준 LDA
칼럼개수        7               8               9

time        248.0194        575.2531        450.5493

acc         0.8888          0.9074          0.9114

'''
