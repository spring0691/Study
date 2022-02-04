from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os

#설계방향
#남자사진이 1418장 여자사진이 1912장 있다. 우리는 증폭을 배웠으니까 남자사진2000장 여자사진2000장으로 맞춘후 모델을 돌려보고싶다.

#1. 데이터 로드 및 전처리

path = '../../_data/image/men_women/'      

#남자사진,여자사진의 이름을 리스트로 불러와보자.
men = os.listdir(path+'men')          # 1418장
women = os.listdir(path+'women')      # 1912장
#print(men[:3])      # ['00000001.jpg', '00000002.jpg', '0000000296.png']   이런식으로 저장되어있다.
#print(women[:3])    # ['00000000.jpg', '00000001.jpg', '00000002.jpg']     이름을 저장할때 index방식이 달라서 순서가 살짝 뒤틀린다.

