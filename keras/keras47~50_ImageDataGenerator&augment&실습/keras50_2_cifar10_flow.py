from typing import Tuple
from tensorflow.keras import datasets
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action='ignore')

#1.데이터 로드 및 전처리.

(x_train, y_train),(x_test,y_test) = cifar10.load_data()
# print(x_train.shape,y_train.shape)     #(50000, 32, 32, 3) (50000, 1)    5만장의 사진과 그에 대한 답입니다~
# print(x_test.shape,y_test.shape)       #(10000, 32, 32, 3) (10000, 1)

#print(np.unique(y_train,return_counts=True))   0가지 종류가 각각 5000장씩들어가있다.

#그림을 직접 한번 보고 가겠습니다.
# plt.figure(figsize=(10,10))
# for i in range(30):
#     plt.subplot(8,8,i+1)
#     plt.imshow(x_test[i])
# plt.show()

#이 데이터는 자동차,전투기,강아지,말,개구리 등등 전혀상관없는 여러 객체의 img를 담고 있습니다.


train_augment_datagen = ImageDataGenerator(    
    #rescale=1./255,       
    horizontal_flip=True,  
    rotation_range=3,       
    width_shift_range=0.3, 
    height_shift_range=0.3, 
    zoom_range=(0.3),       
    fill_mode='nearest',                   
)
all_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

# 지금부터 증폭을 해보겠습니다. 

augment_size = 100000 - x_train.shape[0]    # 증폭할 데이터의 개수 지정 총 10만개해주기 위해서 50000개 지정.
                                            #요거는 5만장이라서 딱 2배하면 되므로 랜덤하게 뽑지말고 5만장 그대로 1장씩 변환 ㄱㄱ

x_augmented = x_train.copy()      
y_augmented = y_train.copy() 
#print(len(x_augmented),len(y_augmented))   # 50000 50000

before_x_augmented = x_augmented.copy()

x_augmented = train_augment_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size, shuffle=False, 
).next()[0]     

after_x_augmented = x_augmented.copy()

#print(after_x_augmented[0][0]) 
'''
 [159.99702  127.997025  96.997025]
 [148.8631   117.10439   86.10439 ]
 [138.37119  107.37119   76.37119 ]  
 왜지 값들이 정수가 아니라 부동소수점으로 나오지? 변환하면서 여러 값들이 먹여지는데 이게
 각각의 픽셀값들을 소수점으로 넘겨줘버리는거 같다. -> 처음부터 rescale해서 작업하고 나중에 추가할때 dategen따로해주자.
'''

'''
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(before_x_augmented[i])
plt.show()       

plt.figure(figsize=(10,10))

for i in range(10):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(after_x_augmented[i])
    #plt.imshow((after_x_augmented[i] * 255).astype(np.uint8))    
plt.show()   
'''

'''
#하... 한번에 2줄로 깔끔하게 변환 전후 출력되게 하고싶은데 숙제때문에 일단 접어두도록 하겠습니다...

real_x_train = np.concatenate((x_train, x_augmented))   
real_y_train = np.concatenate((y_train, y_augmented))
#print(len(real_x_train),type(real_x_train))        #100000 <class 'numpy.ndarray'>
#print(len(real_y_train),type(real_y_train))        #100000 <class 'numpy.ndarray'>

#이제야 비로소 진짜 x_train과 y_train이 10만개씩 만들어졌으므로 이미지제너레이터를 새로해서 작업해보도록하겠습니다.

xy_train_train = all_datagen.flow(
    real_x_train,real_y_train,
    batch_size=100,shuffle=True,seed=66,
    subset='training'
)#.next()[0] 
xy_train_val = all_datagen.flow(
    real_x_train,real_y_train,
    batch_size=100,shuffle=True,seed=66,
    subset='validation'
)#.next()[0]
xy_test = all_datagen.flow(
    x_test,y_test,
    batch_size=100
)#.next()[0]
# all_datagen에 validation_split써놓긴 했지만 subset으로 딱딱 명시해줘야 작동하지 굳이 명시 안하면 스킵하고 잘 작동한다.
#print(len(xy_train_train),len(xy_train_val),len(xy_test)) #80000 20000 10000
#flow 그냥 하는것과 .next() .next()[0]의 차이
#그냥 할 경우
# 80000 길이
# <class 'keras.preprocessing.image.NumpyArrayIterator'>
# 2
# <class 'tuple'>
# 1
# <class 'numpy.ndarray'>

#.next()
# 2 길이
# <class 'tuple'>
# 1
# <class 'numpy.ndarray'>
# 28
# <class 'numpy.ndarray'>

#.next()[0]     [0]을 쓰면 x값만 가져오겠다는 의미 
# 1 길이 
# <class 'numpy.ndarray'>
# 28
# <class 'numpy.ndarray'>
# 28
# <class 'numpy.ndarray'>
# keras49_1_flow에 적은내용이 그대로 나타난다. next() 메소드는 선택한 요소의 바로 다음에 위치한 형제 요소를 선택한다.
# 클래스를 가진 요소의 바로 다음 형제 요소 하나를 선택하여, 해당 요소의 CSS 스타일을 변경한다.


#2. 모델링

model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3),strides=1,padding='valid', input_shape=(32,32,3), activation='relu')) # 30,30,10
model.add(MaxPooling2D(2,2))                                                                                # 15,15,10
model.add(Conv2D(10,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       # 15,15,10
model.add(MaxPooling2D(3,3))                                                                                #  5,5,10
model.add(Conv2D(10,(2,2), activation='relu'))                                                              #  4,4,10
model.add(MaxPooling2D(2,2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit_generator(xy_train_train,epochs=10000,steps_per_epoch=len(xy_train_train)//2,validation_data=xy_train_val, validation_steps=len(xy_train_val),callbacks=[es])

#4. 평가 예측
loss = model.evaluate_generator(xy_test, steps=len(xy_test))
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_pred = model.predict_generator(x_test)

y_pred_int = np.argmax(y_pred,axis=1)        

acc = accuracy_score(y_test,y_pred_int)

print('acc_scroe : ',acc)
'''


'''
# 현재        증폭시키기전 값.
# loss :      1.012665867805481       
# accuracy :  0.6509000062942505        

# train 60000 -> 100000만으로 증폭시킨 후 .
# loss :      
# accuracy :  
# acc_score:                  
'''
