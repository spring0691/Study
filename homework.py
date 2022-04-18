from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D,\
                                    GlobalAveragePooling2D,BatchNormalization,LayerNormalization
import numpy as np
from tensorflow.keras.datasets import cifar100 # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.optimizers import Adam,Adadelta
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#1 데이터 정제작업 !!
datasets = load_breast_cancer()
x = datasets.data           
y = datasets.target         

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)

#2.모델링
model = Sequential()
model.add(Dense(128,activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu',kernel_initializer='he_normal'))
model.add(Dense(2, activation='softmax'))


#3.컴파일, 훈련
optimizer = Adam(learning_rate=0.0001)  # 1e-4     
lr=ReduceLROnPlateau(monitor= "val_acc", patience = 3, mode='max',factor = 0.1, min_lr=1e-6,verbose=False)
es = EarlyStopping(monitor="val_acc", patience= 5, mode='max',verbose=1,baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras33_cifar100_MCP.hdf5')
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['acc']) 
model.fit(x_train,y_train,epochs=100, batch_size=8,validation_split=0.2, callbacks=[lr,es])#


#4.평가,예측
y_pred = model.predict(x_test)
y_pred_int = np.argmax(model.predict(x_test),axis=1)
cf = confusion_matrix(y_test,y_pred_int)
print(cf)