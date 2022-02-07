from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np,pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))

#1. 데이터 로드 및 정제
path = "./_data/bike/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                   
submit_file = pd.read_csv(path + 'sampleSubmission.csv')