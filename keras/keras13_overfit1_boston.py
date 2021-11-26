# 과적합 예제

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 데이터 정제작업 !!
datasets = load_boston()
x = datasets.data
y = datasets.target
'''
print(x)    # x내용물 확인
print(y)    # y내용물 확인
print(x.shape) # x형태
print(y.shape) # y형태
print(datasets.feature_names) # 컬럼,열의 이름들
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
'''

