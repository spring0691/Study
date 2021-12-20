import pandas as pd

ss = pd.read_csv("./삼성전자.csv", thousands=',', encoding='CP949')
ss = ss.drop(range(20, 1120), axis=0)
ki = pd.read_csv("./키움증권.csv", thousands=',', encoding='CP949')
ki = ki.drop(range(20, 1060), axis=0)

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 인덱스 재배열
ss = ss.loc[::-1].reset_index(drop=True)
ki = ki.loc[::-1].reset_index(drop=True)

# 필요한 컬럼만 남겨두기
x_ss = ss.drop(['일자', '고가', '저가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ss = np.array(x_ss)
x_ki = ki.drop(['일자', '고가', '저가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ki = np.array(x_ki)

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_ssp, y_ssp = split_xy(x_ss, 4, 2)
x_kip, y_kip = split_xy(x_ki, 4, 2)

print(x_ssp[-3:])
print(y_ssp[-3:])