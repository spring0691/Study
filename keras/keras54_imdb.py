from tensorflow.keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)
#print(np.unique(y_train))          # [0 1] 감정분석
#print(x_train[0],y_train[0])       # [4,6,6,7,4,6,4,6,43,6,6,43,6,3,43,3,6,7,87] 이런식에 문장이랑 그에 대한 값 1


'''
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

word_to_index = imdb.get_word_index()   # dict
#print(sorted(word_to_index.items()))    #딕셔너리는 key와 values가 있는데 이렇게 하면 key들이 나온다.
import operator # 이 기능을 import해서 retuers자료형이 가지고 있는 모든 어절들을 순.서.대.로. 확인할수있다 
# 'The'가 1번 , 'of'가 2번, ..... '뭐뭐뭐'가 30979번 이런식으로 
# print(sorted(word_to_index.items(), key=operator.itemgetter(1)))    
# sorted오름차순 word_to_index.items()로 값들을 불러오고 key=operator.itemgetter(1)로 키와 밸류중 밸류를 선택해서 오름차순으로 프린트한다.

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token 

print(' '.join([index_to_word[index] for index in x_train[1]]))
'''