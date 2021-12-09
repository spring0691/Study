from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data           # 원본데이터는 나중에 어떻게 쓸지 모르니까  그냥 둠
y = datasets.target         

'''
print(x.shape) # x형태  (506,13)    -> (506,13,1,1) 해서 cnn으로 모델링
print(y.shape) # y형태  (506, )
print(datasets.feature_names) # 컬럼,열의 이름들
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
'''

# cnn 만들기
# img           (1000, 32,32,3) -> (1000, 3072) -> 4차원에서 2차원으로변환
# 2차원         (1000, 3072)  -> (1000, 32,32,3) -> 2차원에서 4차원으로 형태변환 한후 conv2D

# numpy pandas로 변환후 pandas의 제공기능인 index정보와 columns정보를 확인할수있다.
xx = pd.DataFrame(x, columns=datasets.feature_names)    # x가 pandas로 바껴서 xx에 저장, columns를 칼럼명이 나오게 지정해준다.
#print(type(xx))         # pandas.core.frame.DataFrame
#print(xx)               # 잘 되었나 확인.

#print(xx.corr())        # 칼럼들의 서로서로의 상관관게를 도표로 확인할 수 있다.    절대값클수록 양 or 음의 상관관계 0에 가까울수록 서로 영향 없음

xx['price'] = y         # xx의 데이터셋에 y값을 price라는 이름의 칼럼으로 추가한다. 원본데이터는 그대로있다.    열 추가하는 방법.

#print(xx)

# print(xx.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns   # 조금 더 이쁘게 만들게 도와줌.

# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# # seaborn heatmap 개념정리

# plt.show()

xx= xx.drop(['CHAS','price'], axis=1)    # x데이터에서 CHAS열 제거

#print(xx)


#2.모델링

model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3),strides=1,padding='valid', input_shape=(2,2,3), activation='relu'))
model.add(MaxPooling2D(2,2))                                                                                
model.add(Conv2D(10,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       
model.add(MaxPooling2D(3,3))                                                                                
model.add(Conv2D(10,(2,2), activation='relu'))                                                              
model.add(MaxPooling2D(2,2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(1))




