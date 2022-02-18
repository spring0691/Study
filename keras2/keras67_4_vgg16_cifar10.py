from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time

(x_train,y_train), (x_test,y_test) = cifar10.load_data()

# print(x_train.shape)              # 32,32,3
# print(len(np.unique(y_test)))     # 100

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

vgg16.trainable = False     # 가중치를 동결시킨다!

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

# model.trainable = False


optimizer = Adam(learning_rate=0.1)
lr=ReduceLROnPlateau(monitor= "val_acc", patience = 5, mode='max',factor = 0.5, min_lr=0.0001,verbose=1)
es = EarlyStopping(monitor ="val_acc", patience=50, mode='max',verbose=1,restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')

start = time.time()
model.fit(x_train,y_train,batch_size=5,epochs=10000,validation_split=0.2,callbacks=[lr,es])#,cp
end = time.time()
loss, Acc = model.evaluate(x_test,y_test,batch_size=5)
print('loss : ',round(loss,4))
print('Acc : ',round(Acc,4))

# 결과 비교
# vgg trainable : True / False
# Flatten / Global Average Pooling
# 위 4개 조합해서 최고결과 뽑고 이전 최고치 acc0.65와 비교

# 출력결과     True/Flat    True/GAP    False/Flat      False/GAP
# time : 
# loss : 
#  acc :  