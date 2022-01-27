from itertools import groupby
import pandas as pd, numpy as np,sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,PowerTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

# 출력 관련 옵션
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',100)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 150)

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)  # (4898, 12) index와 header은 default값.



