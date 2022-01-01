from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

#1. 데이터 로드 및 전처리

path = '../_data/image/men_women'

men = os.listdir(path+'/men')          # 1418장
women = os.listdir(path+'/women')      # 1912장

#증폭배웠으니까 각각 증폭시켜서 2000장맞춰서 가자.
#일단 이미지를 numpy형태로 변환 후, 이미지제너레이트해야 증폭까지 가능하다.

m = []
for i in men:
    m.append(np.array(Image.open(f'{path}/men/{i}').convert('RGB').resize((300,300))))    #반복문 써서 1418장 변환
# Image.open으로 이미지를 불러오고 컬러형태이기때문에 convert로 RGB계산해서 불러오고 300,300으로 사이즈 조정해준다.
mm = np.array(m)
w = []
for i in women:
    w.append(np.array(Image.open(f'{path}/women/{i}').convert('RGB').resize((300,300))))  #반복문 써서 1912장 변환
ww = np.array(w)

mw_augment_datagen = ImageDataGenerator(        # 남녀 사진 변환용 
    rescale=1./255.,       
    horizontal_flip=True,  
    rotation_range=3,       
    width_shift_range=0.3, 
    height_shift_range=0.3, 
    zoom_range=(0.3),       
    fill_mode='nearest',  
)
all_datagen = ImageDataGenerator(               # 2000장 된 후 쓸거
    rescale=1./255.
)

m_augmented_size = 2000 - len(mm)        # 582개
w_augmented_size = 2000 - len(ww)
m_randidx = np.random.randint(len(mm),size=m_augmented_size) # 1~1418개중에서 임의로 582개의 숫자 뽑아서 저장
w_randidx = np.random.randint(len(ww),size=w_augmented_size)
m_augmented = mm[m_randidx] # 남자 변환할 사진들을 mm에서 가져옴
w_augmented = ww[w_randidx]

m_augmented = mw_augment_datagen.flow(
    m_augmented, np.zeros(m_augmented_size),
    batch_size=m_augmented_size, shuffle=True,seed=66
).next()[0] # 변환 후 다시 x값만 저장.
w_augmented = mw_augment_datagen.flow(
    w_augmented, np.zeros(w_augmented_size),
    batch_size=w_augmented_size, shuffle=True,seed=66
).next()[0]
#print(len(m_augmented),len(w_augmented))  # 582 88  드디어 각각 변환 완료... 다시 추가해서 리스케일 할거기 때문에 x255해준다.
m_augmented = m_augmented * 255.
w_augmented = w_augmented * 255.

real_men = np.concatenate((mm,m_augmented))
real_women = np.concatenate((ww,w_augmented))
#print(len(real_men),len(real_women))       # 2000 2000 각각 2000장씩 맞춤. 

#남녀 사진을 각각 다시 폴더 만들어서 train_set안에 남녀 따로 test_set안에 남녀 따로 해서 flow_from_diretory하거나
# flow로 men women각각 만들어주고 앙상블모델로 돌린다. -> 앙상블 모델로 못돌린다 남녀 엮어서 1 0 이렇게 세트로 해야하는데 
# 앙상블로 엮었다고 해도 output을 
#폴더 만들어서 하는건 keras48_2_hores_or_human_IDG여기에서 해봤으니까 지금은 flow하고 앙상블해보자.

#0에 가까울수록 여자 1에 가까울수록 남자 
w = all_datagen.flow(
    real_women,np.zeros(len(real_women)),
    batch_size=len(real_women),shuffle=True,seed=66
)
m = all_datagen.flow(
    real_men,np.ones(len(real_men)),
    batch_size=len(real_men),shuffle=True,seed=66
)
# print(len(w),len(m))    # 2000 2000 잘 나눠진거 확인. 이제 xy나눠줘야함.
# print(type(m))          # <class 'keras.preprocessing.image.NumpyArrayIterator'>
# print(type(m[0]))       # <class 'tuple'>
# print(type(m[0][0]),type(m[0][1]))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(len(m[0][0]),len(m[0][1]))    # 1 1
# print(w[0][0],w[0][1])      #  .......[0.9490197  0.95294124 0.9333334 ]]]] [0.]
# print(m[0][0],m[0][1])      #  .......[0.53333336 0.56078434 0.65882355]]]] [1.]

# 이미지 1장 당 0과 1값이 매겨져 있는것 확인. 
# 이제 넘파이형태로 저장해서 다음 작업부터 좀 빠르게 불러오자.

np.save(f'{path}/women_x.npy', arr=w[0][0])
np.save(f'{path}/women_y.npy', arr=w[0][1])
np.save(f'{path}/men_x.npy', arr=m[0][0])
np.save(f'{path}/men_y.npy', arr=m[0][1])       # 드디어 저장까지 끝났다 이제부터는 로드해서 사용하자.


women_x = np.load(f'{path}/women_x.npy')
women_y = np.load(f'{path}/women_y.npy')
men_x = np.load(f'{path}/men_x.npy')
men_y = np.load(f'{path}/men_y.npy')

# print(women_x.shape,women_y.shape,men_x.shape,men_y.shape)     # 2000장 확인완료!
# print(women_x[0],women_y[0])      #0확인
# print(men_x[0],men_y[0])          #1확인

# 마지막으로 사진으로 한번만 확인하고 가자.
# plt.figure(figsize=(10,10))
# for i in range(10):
#     plt.subplot(2,10,i+1)
#     plt.axis('off')
#     plt.imshow(men_x[i])
#     plt.subplot(2,10,i+11)
#     plt.axis('off')
#     plt.imshow(women_x[i])
# plt.show()                            두 줄 출력 완료.

# train test분리
#women_x_train, women_x_test, women_y_train, women_y_test = train_test_split(women_x,women_y,train_size=0.8,shuffle=True,random_state=66)
#men_x_train, men_x_test, men_y_train, men_y_test = train_test_split(men_x,men_y,train_size=0.8,shuffle=True,random_state=66)
# print(len(women_x_train),len(women_x_test),len(women_y_train),len(women_y_test))      # 1600 400 1600 400
# print(len(men_x_train),len(men_x_test),len(men_y_train),len(men_y_test))              # 1600 400 1600 400

#2. 모델링

def build_model(maxpool=0.5,dropout=0.5,optimizer='adam', activation='relu'):
    winputs = Input(shape=(300,300,3), name='winputs')
    x = Conv2D(128,kernel_size=(2,2),padding='same')(winputs)
    x = MaxPool2D(maxpool)(x)
    x = Conv2D(64,kernel_size=(2,2),padding='same')(x)
    x = MaxPool2D(maxpool)(x)
    x = Flatten()(x)
    x = Dense(512,activation = activation,name='hidden1')(x)
    x = Dropout(dropout)(x)
    x = Dense(256,activation = activation,name='hidden2')(x)
    x = Dropout(dropout)(x)
    x = Dense(128,activation = activation,name='hidden3')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1,name='wouputs')(x)
    
    
    
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer=optimizer,metrics=['acc'],loss='categorical_crossentropy')
    return model

# 함수형으로 만든 하이퍼 파라미터

def  create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop','adam','adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4]
    activations = ['relu','tanh','sigmoid']
    return {'batch_size' : batches, 'optimizer' : optimizers, 'drop': dropout, 'activation' : activations}

hyperparameters = create_hyperparameters()
model2 = build_model

# 그냥 모델을 서치에 넣으면 안된다 랩핑안에 넣어서 돌려야 한다
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model,verbose = 1)
#  여기까지가 랩핑

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3)

search.fit(x_train,y_train,verbose=1)

print(search.best_params_) # 내가 선택한 세개의 파라미터 {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}
print(search.best_estimator_) # 모든 파라미터에 대한 것(내가 튠하지 않은것도 나온다) 랩핑한거라 뒤 처럼 나온다 <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000023E8D35B670>
print(search.best_score_) # 0.9561999837557474 스코어랑 수치가 다르다

acc = search.score(x_test,y_test) # acc :  0.9581999778747559
print('acc : ', acc)