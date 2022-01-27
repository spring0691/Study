# 실습 . 
# 3,4 -> 0      5,6,7 -> 1      8,9 ->2      
# 3,4,5 -> 0        6 -> 1      7,8,9 ->2

# 각각 증폭 전/후 해서 라벨축소 전/후 해서 -> 8가지 결과
# 원본, 원본&label분포증폭, 원본&라벨범위축소 X 2, 

from tabnanny import verbose
import numpy as np, pandas as pd, warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.metrics import f1_score,accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings(action='ignore')

#1. 데이터

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)
x = datasets.drop(['quality'],axis=1) 
y = datasets['quality']  

# le = LabelEncoder()
# y = le.fit_transform(y)

# print(np.unique(y,return_counts=True))  # [3, 4, 5, 6, 7, 8, 9] [20, 163, 1457, 2198, 880, 175, 5]

# label축소 case1,2
# y= y.apply(lambda x: 0 if x <= 4 else 1 if x<= 7 else 2)  # [0, 1, 2]   [ 183, 4535, 180]
y= y.apply(lambda x: 0 if x <= 5 else 1 if x == 6 else 2) # [0, 1, 2]   [1640, 2198, 1060]

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y) 

# smote
smote = SMOTE(random_state=66,k_neighbors=5)    
x_train,y_train = smote.fit_resample(x_train,y_train)
             
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66,stratify=y_train)  

model = XGBClassifier(
    n_estimators = 1000,
    max_depth = 9,
    learning_rate = 0.05,
    reg_lambda = 1,
    reg_alpha = 1,
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor',
    eval_metric='merror',
    use_label_encoder=False,
)

model.fit(x_train,y_train,verbose=True,early_stopping_rounds=100,eval_set=[(x_val,y_val)])

print(f"md.score : {round(model.score(x_test,y_test),4)}")                             
print(f"ac_score : {round(accuracy_score(y_test,model.predict(x_test)),4)}")            
print(f"f1_score : {round(f1_score(y_test,model.predict(x_test),average='macro'),4)}")   
print(f"f1_score : {round(f1_score(y_test,model.predict(x_test),average='micro'),4)}") 

'''
축소 x    smote x
md.score : 0.6388
ac_score : 0.6388
f1_score : 0.3698
f1_score : 0.6388

축소 x   smote3
md.score : 0.6112
ac_score : 0.6112
f1_score : 0.3785
f1_score : 0.6112

축소1 o   smote x
md.score : 0.7122
ac_score : 0.7122
f1_score : 0.7088
f1_score : 0.7122

축소1 o   smote5
md.score : 0.7143
ac_score : 0.7143
f1_score : 0.7142
f1_score : 0.7143

축소2 o  smote x
md.score : 0.9357
ac_score : 0.9357
f1_score : 0.5364
f1_score : 0.9357

축소2 o  smote 4
md.score : 0.9051
ac_score : 0.9051
f1_score : 0.5958
f1_score : 0.9051
'''