from tensorflow.keras.datasets import reuters
import numpy as np,pandas as pd

(x_train,y_train), (x_test,y_test) = reuters.load_data(
    num_words=5000, test_split=0.2     # 단어사전의 개수 단어의 개수
)
#print(x_train, len(x_train))   # 8982, 2246
#print(y_train[0])               # 3
#print(np.unique(y_train))       # 46개의 labels종류    
#reuters 자료형은 뉴스의 카테고리   총 11228개의 뉴스를 가져와서 46개의 카테고리로 나누었다.

#print(type(x_train),type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
#print(x_train.shape,y_train.shape)  # (8982,) (8982,)
#print(len(x_train[0]),len(x_train[1]))  # 87,56 
#print(type(x_train[0]), type(x_train[1])) #<class 'list'> <class 'list'>
#이게 무슨 말이냐 리스트들을 감싸서 넘파이로 만들었다.

#print('뉴스기사의 최대길이 : ', max(len(i) for i in x_train ))                 #  2376
# for i in x_train으로 x_train안에서 반복하면서 8982개의 뉴스기사들의 길이를 하나씩 찍어본다. 8982개의 뉴스기사들의 길이list에서 제일 큰 값 반환(max)
#print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train)   )      #  145.5398574927633
# map(len, x_train) x_train의 길이만큼 반복하면서 len을 하나씩 다 반환해준다 sum()으로 다 더해준 후 개수(8982)로 나눠줌.
# map(len, x_train)) 과  len(i) for i in x_train 가 같은 기능을 수행한다.

# 전처리 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

x_train = pad_sequences(x_train, padding='pre', maxlen=140, truncating='pre')   # 100개의 어절보다 작으면 0 채우고 크면 앞에서부터 잘라서 100개를 만들겠다.
x_test = pad_sequences(x_test, padding='pre', maxlen=140, truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x_train.shape,y_train.shape)  # (8982, 140) (8982, 46)
#print(x_test.shape,y_test.shape)    # (2246, 140) (2246, 46)

'''
# sort 오름차순... 이거 파이썬 기초때 베움... 근데 까먹고있었네
# word_to_index = reuters.get_word_index()
# print(type(word_to_index))
# print(sorted(word_to_index.items()))    딕셔너리는 key와 values가 있는데 이렇게 하면 key들이 나온다.
# import operator # 이 기능을 import해서 retuers자료형이 가지고 있는 모든 어절들을 순.서.대.로. 확인할수있다 
# 'The'가 1번 , 'of'가 2번, ..... '뭐뭐뭐'가 30979번 이런식으로 
# print(sorted(word_to_index.items(), key=operator.itemgetter(1)))    
# sorted오름차순 word_to_index.items()로 값들을 불러오고 key=operator.itemgetter(1)로 키와 밸류중 밸류를 선택해서 오름차순으로 프린트한다.

# index_to_word = {}
# for key, value in word_to_index.items():
#     index_to_word[value+3] = key

# for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
#     index_to_word[index] = token 

# print(' '.join([index_to_word[index] for index in x_train[1]]))
'''

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Embedding(5001,50,input_length=140))   
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(46,activation='softmax'))


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor="val_acc", patience=100, mode='max',verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000,batch_size=100,validation_split=0.2,callbacks=[es])

#4. 평가,예측

acc = model.evaluate(x_test,y_test,batch_size=100)

print('loss & acc : ', acc)

y_pred = model.predict(x_test)

y_test_int = np.argmax(y_test, axis=1)
y_pred_int = np.argmax(y_pred, axis=1)

acc_score = accuracy_score(y_test_int,y_pred_int)
print('acc_score : ', acc_score)



'''
acc와 acc_score가 같아야 정상. 사실 다를수가없음;
num_word   5000         10000           20000
loss       4.3311       4.1296          8.34843
acc:       0.6224       0.6340          0.58637
acc_scroe: 0.6224
'''