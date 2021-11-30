from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from pandas import get_dummies
import numpy as np
import pandas as pd     # csv같은 데이터를 분석하는걸 도와주는 api
# csv 거의 엑셀이랑 비스사다 데이터의 구분을 ,로 해준다.
# 확장에서 rainbow csv설치 - csv파일을 보기좋게 색같은거넣어서 구분해준다. 
# edit csv설치    - csv파일을 edit창에서 열어서 엑셀처럼 더 편하게 보기좋게 해준다.

#1. 데이터 로드 및 정제
path = "./_data/titanic/"
train = pd.read_csv(path + "train.csv", index_col=0, header=0)  # index_col=0... 열을 어디부터 읽을거냐 header=0.. 행을 어디부터 읽을거냐
test = pd.read_csv(path +"test.csv",index_col=0, header=0)
gender_submission = pd.read_csv(path +"gender_submission.csv",index_col=0, header=0)    #제출용 파일 여기에 값을 덮어쓴다.
#print(train.shape)                 #(891, 11)
#print(test.shape)                  #(418, 10)         
#print(gender_submission.shape)     #(418, 1)
#print(train.info())
#print(train.describe())    object형은 연산을 못해서 결과에서 빠져있다. 6개의  colums만 나온다.

# test데이터는 칼럼이 하나빠져있다. train으로 모델만들고 예상해서 test에 결과값행을 하나 추가해서 test를 완성시킨다.
# gender_submission에 최종완성시켜서 제출하는거다.
# 모든 대회에서 train데이터 가지고 훈련시켜서 test를 완성시켜보고 그걸 가다듬에서 최종제출함.

# 못배운진도랑 기능이 많아서 일단 보류  결측치 이상치 string를 label로 바꿔주는 거 모름 .