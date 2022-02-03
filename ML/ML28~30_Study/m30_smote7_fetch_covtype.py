# value_counts() 해서 라벨값 한번 보고 upsampling해서 균형으로맞추고 작업해라
# 지금까지 배운 저장 기능을 활용하여. upsampling한 데이터 저장하고 
# 데이터를 저장 한 후 load하여 사용해라

import numpy as np, pandas as pd,warnings
# from sklearn.datasets import fetch_covtype
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from xgboost import XGBClassifier
from xgboost import cv
from lightgbm import LGBMClassifier
from lightgbm import CVBooster
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder

#출력관련 옵션
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_row',100)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 180)

path = 'D:\_data\\fetch_covtype'

# <-----------------------------------------------------------------------------------------------------  저장영역
'''
datasets = fetch_covtype()
x = datasets.data
y = datasets.target # [1, 2, 3, 4, 5, 6, 7] [211840, 283301, 35754, 2747, 9493, 17367, 20510]

#절대 영역 - 여기서부터 smote까지는 k_neighbors의 값 차이만 있지 절대적인 데이터. 저장 후 load
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8,stratify=y)

np.save(f"{path}/fetch_covtype_x_test.npy",x_test)
np.save(f"{path}/fetch_covtype_y_test.npy",y_test)

k_num_list = [2,4,6,8,10]

for k_num in k_num_list:
    smote = SMOTE(random_state=66,k_neighbors=k_num,n_jobs=-1)    
    x_train,y_train = smote.fit_resample(x_train,y_train)

    np.save(f"{path}/fetch_covtype_x_train{k_num}.npy",x_train)
    np.save(f"{path}/fetch_covtype_y_train{k_num}.npy",y_train)
'''
#<------------------------------------------------------ 이 시점에서 k_neighbors count별로 x_train,y_train 저장해줘야함.

#<------------------------------------------------------------------------ 데이터 load

x_train = np.load(f'{path}/fetch_covtype_x_train6.npy') 
y_train = np.load(f'{path}/fetch_covtype_y_train6.npy') 
x_test = np.load(f'{path}/fetch_covtype_x_test.npy') 
y_test = np.load(f'{path}/fetch_covtype_y_test.npy') 

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# print(np.unique(y_train,return_counts=True))    # [1, 2, 3, 4, 5, 6, 7] [226640, 226640, 226640, 226640, 226640, 226640, 226640]
# print(np.unique(y_test,return_counts=True))     # [1, 2, 3, 4, 5, 6, 7] [42368, 56661,  7151,   549,  1899,  3473,  4102]
# 제일 큰 값 끼리 더해보면 283301

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66,stratify=y_train) 

# parameters = {"learning_rate":[0.025,0.05,0.075,0.01],"max_depth":[4,6,8,10],"reg_alpha":[0,1],"reg_lambda":[0,1],
#               "n_estimators":[200,300,400]}

# ,"device":['gpu'],"min_child_weight":[0.5, 1],"subsample":[0.5,1],"colsample_bytree":[0.5,1]
# 구현 못한 earlystop & randomsearch 나머지 코드
#"early_stopping_rounds":[100],"eval_set":[(x_val,y_val)],
# ** 파라미터 경우의 수 4 * 4 * 2 * 2 * 3   

# model = RandomizedSearchCV(
#             XGBClassifier(tree_method = 'gpu_hist',predictor = 'gpu_predictor',eval_metric='mlogloss',use_label_encoder=False),  # ,eval_metric='merror'
#             parameters,cv=3,verbose=2,n_jobs=-1,random_state=66,n_iter=100, refit=True)

# model = XGBClassifier( 
#     tree_method = 'gpu_hist',predictor = 'gpu_predictor',eval_metric='merror',use_label_encoder=False,
#     learning_rate = 0.075, max_depth = 15, reg_alpha = 0, reg_lambda = 1, n_estimators=100000
# )

# model.fit(x_train,y_train,verbose=True,early_stopping_rounds = 100,eval_set = [(x_val,y_val)])

# model.save_model(path + "/m30_covtype_xgb_model6.dat")
model = XGBClassifier()
model.load_model(path + "/m30_covtype_xgb_model4.dat")

# print(f"최적의 파라미터 : {model.best_params_}\n")  
print(f"md.score : {model.score(x_test,y_test)}")                               # 0.9485727562971696        
print(f"ac_score : {accuracy_score(y_test,model.predict(x_test))}")             # 0.9485727562971696
print(f"f1_score : {f1_score(y_test,model.predict(x_test),average='macro')}")   # 0.9383842703316215

# 1번 모델 learning_rate = 0.025, max_depth = 6, reg_alpha = 0, reg_lambda = 1, acc 0.9485727562971696  f1 0.9383842703316215   save X
# 2번 모델 learning_rate = 0.05, max_depth = 8, reg_alpha = 0, reg_lambda = 1   acc 0.9657668046435979  f1 0.9481580625488778   save X 실수로 덮어씀
# 3번 모델 learning_rate = 0.05, max_depth = 10, reg_alpha = 1, reg_lambda = 0  acc 0.9689767045601232  f1 0.9502983620158572   
# 4번 모델 learning_rate = 0.075, max_depth = 12, reg_alpha = 1, reg_lambda = 0 acc 0.9697339999827888  f1 0.951052157943805
# 5번 모델 learning_rate = 0.01, max_depth = 12, reg_alpha = 0, reg_lambda = 1  acc 0.9661368467251276  f1 0.9484070997522328

# 새로 best_params값으로 한번 다시돌려야 뽑을수있음 절대 이어서 못함.
# print(f"FImports : {model.feature_importances_}")
