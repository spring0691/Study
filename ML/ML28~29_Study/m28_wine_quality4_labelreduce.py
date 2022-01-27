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

x = datasets.drop(['quality'],axis=1)     
y = datasets['quality']  

# 컬럼이 너무 많을 필요는 없다. -> 소비자 입장에서 상 중 하 등급정도면 충분하지 굳이 7등급 다 따지지 않는다. 매우중요! 권한이 있을때 해주자
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64)) 
# 묶어주자!

# case1. 내 방법!
# y= y.apply(lambda x: 1 if x == 3 or x == 4 or x == 5 else 2 if x== 6 else 3)  # apply 적용하다

# case2. 쌤 방법 예전에 내가 하던 방법. for문이용하는거


for i,v in enumerate(y):
    if v == 9:
        y[i] = 8
    elif v == 8:
        y[i] = 2
    elif v == 7:
        y[i] = 7
    elif v == 6:
        y[i] = 6
    elif v == 5:
        y[i] = 5
    elif v == 4:
        y[i] = 4
    elif v == 3:
        y[i] = 3
print(np.unique(y,return_counts=True))

'''
# case3. 소담,명재 형님이 하던 np.where이용하는 방법
y = np.where(y<=5,'Good',np.where(y==6,'Normal',np.where(y<=9,'Bad',y)))
print(np.unique(y,return_counts=True))

le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y)  

model = XGBClassifier(
    tree_method = 'gpu_hist',predictor = 'gpu_predictor',eval_metric='merror',use_label_encoder=False,
    reg_lambda = 1, reg_alpha = 1, n_estimators = 10000, max_depth = 9, learning_rate = 0.05
)

model.fit(x_train,y_train)

print(f"md.score : {model.score(x_test,y_test)}")                      
print(f"ac_score : {accuracy_score(y_test,model.predict(x_test))}")     
print(f"f1_score : {f1_score(y_test,model.predict(x_test),average='macro')}")
'''