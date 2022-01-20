# 실습
# 이전에 구한 pca 0.95 0.99 0.999 1.0의 각 칼럼개수만큼 pca먹이고
# 이전에 만들어놓은 784개 칼럼 다 사용한 DNN모델(최상0.978)와 비교

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np,time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

scaler =MinMaxScaler()   
use_x_train = scaler.fit_transform(x_train)     
use_x_test = scaler.transform(x_test) 

pca_name_dict = {'0.95':154,'0.99':331,'0.999':486,'1.0':713}
pca_result_dict = {}
    
for pca_acc,pca_num in pca_name_dict.items():
      
    pca = PCA(n_components=pca_num)
    x_train = pca.fit_transform(use_x_train)
    x_test = pca.transform(use_x_test)

    #2. 모델링
    input1 = Input(shape=(pca_num,))            
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

    pca_result_dict[f'{pca_acc}_time&acc']=[timee,acc]
    
for key, value in pca_result_dict.items():
        print(key, value)



'''                     DNN 기준
칼럼개수   154(0.95)    331(0.99)   486(0.999)  713(1.0)        784

time       604.8084     211.4133    234.6598    253.9843    152.4186

acc         0.9724       0.9661      0.9635      0.9627      0.9628   

time       382.2638     240.8586    200.3685    201.6814    328.6525

acc        0.9692       0.967       0.9607      0.9596      0.9652

                        CNN 
                        

'''