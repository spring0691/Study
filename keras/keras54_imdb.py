from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=50000)

#print(x_train.shape,len(x_train),x_test.shape,len(x_test))  # (25000,) (25000,)
#print(y_train.shape,len(y_train),y_test.shape,len(y_test))  # (25000,) (25000,)
#print(np.unique(y_train))          # [0 1] 감정분석
#print(x_train[0],y_train[0])       # [4,6,6,7,4,6,4,6,43,6,6,43,6,3,43,3,6,7,87] 이런식에 문장이랑 그에 대한 값 1
#print('뉴스기사의 최대길이 : ', max(len(i) for i in x_train ))                 #  2494
#print('뉴스기사의 평균길이 : ', sum(map(len, x_train)) / len(x_train)   )      #  238.71364

x_train = pad_sequences(x_train, padding='pre', maxlen=238, truncating='pre')   
x_test = pad_sequences(x_test, padding='pre', maxlen=238, truncating='pre')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x가 어떤 내용 담고 있는지 확인해 보기.
imdb_index = imdb.get_word_index()   
#print(len(imdb_index))             #88584  이만큼의 단어들이 있다.
import operator 
#print(sorted(imdb_index.items(), key=operator.itemgetter(1)))  순서대로 단어들 확인해보기.

'''
index_to_word = {}
for key, value in imdb_index.items():
    index_to_word[value+3] = key

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token 

print(' '.join([index_to_word[index] for index in x_train[1]]))
'''

model = Sequential()
model.add(Embedding(50001,50,input_length=238))   
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='softmax'))


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
num_words :   50000
loss :        1.17233
acc :         0.83196
acc_score :   0.83196
'''