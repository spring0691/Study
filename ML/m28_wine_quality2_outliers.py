import pandas as pd, numpy as np,sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,PowerTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import EllipticEnvelope

# 출력 관련 옵션
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',100)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 150)

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)  # (4898, 12) index와 header은 default값.

datasets = datasets.to_numpy()

x = datasets[:, :11]    #(4898, 11)
y = datasets[:, 11]     #(4898,)

################# Outliers 확인 #######################

#1. API이용하여 이상치의 위치를 구하고 그 이상치를 Nan으로 바꾼 후 다시 fillna이용하여 Nan값을 채운다.
outliers = EllipticEnvelope(contamination=.1)
for i in range(11):
    col = x[:,i].reshape(-1,1)
    outliers.fit(col)
    Ol = outliers.predict(col)
    print(Ol)
    # Ol안에서 -1이 있는 index번호와 매칭되는 col의 자리를 Nan으로 바꿈.
    # col에 fillna를 이용하여 보간법해서 새로운 값을 채워줌.
    break

#2. 이전수업에서 만든 함수를 이용하여 

################# Outliers 확인 #######################
'''
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y) # yes의 y가 아니라. y의 y다. 

scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_jobs=-1)

#3. 훈련
model.fit(x_train,y_train,eval_metric='merror')

#4. 평가, 예측
score = model.score(x_test,y_test)
print(f'model.score : {score}')

y_predict = model.predict(x_test)

acc = accuracy_score(y_test,y_predict)
f1= f1_score(y_test,y_predict,average='macro')  # [‘micro’, ‘macro’, ‘samples’,’weighted’ 중 하나 선택]
print(f'acc_score : {acc}')
print(f'f1_score : {f1}')
'''