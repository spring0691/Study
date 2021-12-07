import numpy as np
from tensorflow.keras.datasets import mnist # 교육용데이터 
import matplotlib.pyplot as plt

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1) # <-- reshape의 개념은 다 곱해서 그니까 일렬로 만든후 다시 나눠서 재배열하는 개념
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_train.shape)

print(np.unique(y_train, return_counts=True))   # np.unique(y_train, return_counts=True) 하면 pandas의 value.counts와 같은 기능

# 평가지표 acc 
# 0.98 
# val split해서 test를 평가용으로만 써라
