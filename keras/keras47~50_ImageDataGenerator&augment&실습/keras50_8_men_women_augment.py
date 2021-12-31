from PIL.Image import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

#1. 데이터 로드 및 전처리

path = '../_data/image/men_women'

men = os.listdir(path+'/men')          # 1418장
women = os.listdir(path+'/women')      # 1912장


#증폭배웠으니까 각각 증폭시켜서 2000장맞춰서 가자.
#일단 이미지를 numpy형태로 변환 후, 이미지제너레이트해야 증폭까지 가능하다.

#men_numpy = np.array(Image.open('../_data/image/men_women/men/00000001.jpg'))  1장만 변환

m = []
for i in men:
    m.append(   np.array(Image.open (f'{path}/men/{i}').resize((300,300))))    #반복문 써서 1418장 변환
# w = []
# for i in women:
#     m.append(np.array(Image.open(f'{path}/women/{i}').resize((300,300))))  #반복문 써서 1912장 변환

mw_augment_datagen = ImageDataGenerator(
    rescale=1./255.,       
    horizontal_flip=True,  
    rotation_range=3,       
    width_shift_range=0.3, 
    height_shift_range=0.3, 
    zoom_range=(0.3),       
    fill_mode='nearest',  
)

all_datagen = ImageDataGenerator(
    rescale=1./255.,
    validation_split=0.2 
)

m_augmented_size = 2000 - len(m)        # 582개
m_randidx = np.random.randint(len(m),size=m_augmented_size) # 1~1418개중에서 임의로 582개의 숫자 뽑아서 저장

'''
m_augmented = m[m_randidx]
m_augmented = m_augmented.reshape(m_augmented[0],m_augmented[1],m_augmented[2],1)
print(m_augmented.shape)

m_augmented = mw_augment_datagen.flow(
    m_augmented, np.zeros(m_augmented_size),
    batch_size=m_augmented_size, shuffle=False
).next()[0]

plt.figure(figsize=(7,7))
for i in range(20):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(m_augmented[i])
plt.show() 
'''