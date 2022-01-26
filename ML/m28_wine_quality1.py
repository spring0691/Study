import pandas as pd, numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical     
from sklearn.preprocessing import LabelEncoder

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';')

x = datasets.drop(['quality'],axis=1)   # (4898, 11)
y = datasets['quality']                 # [3, 4, 5, 6, 7, 8, 9]가 각각 [  20,  163, 1457, 2198,  880,  175,    5] 되어있다.

le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y)

model = XGBClassifier(
    objective = 'multi : softmax',
    num_class = -1,
    use_label_encoder=False
)

model.fit(x_train,y_train,eval_metric='mlogloss')

print(model.score(x_test,y_test))


