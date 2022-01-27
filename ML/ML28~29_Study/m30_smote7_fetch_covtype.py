# value_counts() 해서 라벨값 한번 보고 upsampling해서 균형으로맞추고 작업해라
# 지금까지 배운 저장 기능을 활용하여. upsampling한 데이터 저장하고 
# 데이터를 저장 한 후 load하여 사용해라

import numpy as np, pandas as pd,warnings
from sklearn.datasets import fetch_covtype,load_iris
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,f1_score

warnings.filterwarnings(action='ignore')

datasets = load_iris()
x = datasets.data
y = datasets.target # [1, 2, 3, 4, 5, 6, 7] [211840, 283301, 35754, 2747, 9493, 17367, 20510]

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8,stratify=y)

smote = SMOTE(random_state=66,k_neighbors=5)    
x_train,y_train = smote.fit_resample(x_train,y_train)



#<------------------------------------------------------ 이 시점에서 k_neighbors count별로 x_train,y_train 저장해줘야함.  

# x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66,stratify=y_train) 

parameters = {"learning_rate":[0.025,0.05,0.075,0.01],"max_depth":[4,6,8,10],"reg_alpha":[0,1],"reg_lambda":[0,1],
              "min_child_weight":[0.5, 1],"gamma":[0,0.5,1],"subsample":[0.5,1],"colsample_bytree":[0.5,1]}
# ** 파라미터 경우의 수 4 * 4 * 2 * 2 * 2 * 2 * 2 * 2   =   1024

model = RandomizedSearchCV(
            LGBMClassifier(n_estimators=10,tree_method = 'gpu_hist',predictor = 'gpu_predictor',eval_metric='merror'
                          ),  #early_stopping_rounds=100,eval_set=[(x_val,y_val)]
            parameters,cv=5,random_state=66,n_iter=10,verbose=2,n_jobs=-1)

model.fit(x_train,y_train,verbose=2)



print(f"최적의 파라미터 : {model.best_params_}\n")  
print(f"md.score : {model.score(x_test,y_test)}")                              
print(f"ac_score : {accuracy_score(y_test,model.predict(x_test))}")            
print(f"f1_score : {f1_score(y_test,model.predict(x_test),average='macro')}")

# 새로 best_params값으로 한번 다시돌려야 뽑을수있음 절대 이어서 못함.
#print(f"FImports : {model.feature_importances_}")
