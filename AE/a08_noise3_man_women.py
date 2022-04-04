# 과제
# 남자 여자 데이터에 노이즈를 넣어서 기미 주근깨 여드름 제거

import numpy as np, os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

#1. 데이터
path = 'D:\_data\men_women\\real_use'

img_list = os.listdir(path)

img_npy = []

for image in img_list:
    img_npy.append(np.array(Image.open(f'{path}/{image}').convert('RGB').resize((300,300))).astype('float')/255)    
    
img_npy = np.array(img_npy)

img_train,img_test = train_test_split(img_npy,train_size=0.9)

img_train_noised = img_train + np.random.normal(0, 0.1, size = img_train.shape) 
img_test_noised = img_test + np.random.normal(0, 0.1, size = img_test.shape) 
img_train_noised = np.clip(img_train_noised, a_min=0, a_max=1)
img_test_noised = np.clip(img_test_noised, a_min=0, a_max=1)

#2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size,kernel_size=(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size*2,(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size*4,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size*2,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(3,(2,2),padding='same',activation='relu'))
    return model

model = autoencoder(hidden_layer_size=64)

model.compile(optimizer='adam', loss='mae')
lr=ReduceLROnPlateau(monitor= "val_loss", patience = 3, mode='min',factor = 0.1, min_lr=0.00001,verbose=1)
es = EarlyStopping(monitor ="val_loss", patience=5, mode='min',verbose=1,restore_best_weights=True)
model.fit(img_train_noised,img_train, epochs=100,batch_size=10,validation_split=0.2,callbacks=[lr,es])

output = (model.predict(img_test_noised)*255).astype(np.uint8)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15),(ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(3,5,figsize = (20,7))      # \는 줄바꿈 위해

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(img_test[random_images[i]])
    if i == 0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(img_test_noised[random_images[i]])
    if i == 0:
        ax.set_ylabel("NOISED", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# predict한 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]])
    if i == 0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
