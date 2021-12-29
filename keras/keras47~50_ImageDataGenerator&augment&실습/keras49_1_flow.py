from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True,               
    #vertical_flip=True,                                      
    width_shift_range=0.1,            
    height_shift_range=0.1,   
    #rotation_range=5,               
    zoom_range=0.1,                 
    #shear_range=0.7,                    
    fill_mode='nearest'          
)
# 이 옵션들이 되게 중요함. 각 이미지 타입들마다 그 이미지들의 특성이 있기때문에 변환특성을 잘 보고 각 데이터에 맞게 적절한 비율로 커스터마이즈 해줘야함. ****

augment_size = 100
# print(x_train[0].shape)                                                                 # (28, 28)
# print(x_train[0].reshape(28*28).shape)                                                  # (784,)
# print(np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1).shape)        # (100, 28, 28, 1)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),  # 28*28대신 -1도 가능. np.tile 반복한다. 784,라는 데이터를 아큐먼트 사이즈만큼 반복하겠다.
    np.zeros(augment_size),          # y값 100개를 전부다 0으로 넣어주겠다.
    batch_size=augment_size,          # 100개로 쫙 펴진걸 100개씩 묶어줌 -> 사진 1장씩 묶어주겠다.   
    shuffle=False
).next()                             # 데이터 증폭해 주기 위한 일련의 작업.

print(len(x_data))
# print(type(x_data))                         # <class 'tuple'>
# print(x_data[0].shape, x_data[1].shape)     # (100, 28, 28, 1) (100,)   튜플로 묶인 세트의 0과1이므로 곧 x와 y값 의미.

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()                                  # 100배로 불린 사진들을 49개만 판에 출력해보겠다.

# 복습 다시 필요함.