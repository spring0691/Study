from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, SimpleRNN, LSTM, GRU
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, f1_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time

def split_x(dataset, size):                             
    aaa = []                                            
    for i in range(len(dataset) - size + 1):            
        subset = dataset[i : (i + size)]                
        aaa.append(subset)                              
    return np.array(aaa) 