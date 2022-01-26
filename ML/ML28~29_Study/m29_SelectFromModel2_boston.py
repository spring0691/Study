# feature 줄여가면서 최적의 값 뽑아보자. 

import numpy as np, pandas as pd, warnings
from sklearn import datasets
from sklearn.datasets import load_boston
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.feature_selection  import SelectFromModel

warnings.filterwarnings(action='ignore')

#1. 데이터

dataset = load_boston()   # (506, 13) (506,)
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
y = dataset.target
#print(y.min(),y.max()) # 5.0 50.0  요건 따로 log적용 안해줘도 되겠다.

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)  

# scaler = StandardScaler()
# real_x_train = scaler.fit_transform(x_train)
# real_x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(tree_method = 'gpu_hist',predictor = 'gpu_predictor')

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
score = model.score(x_test,y_test)
print(f'model.score : {np.round(score,4)}\n')         # defalut + no scaler 0.9142

Fi = pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=x.columns)#.sort_values(by=0,axis=1)
# print(f'Feature_Inportances\n{Fi}\n')
# print(f'Feature_Inportances_sort\n{Fi.sort_values(by=0,axis=1)}')
aaa = np.sort(model.feature_importances_)
# [0.00351779 0.00429927 0.00659664 0.00900961 0.01379713 0.0148002 0.0213099  
# 0.02666957 0.03820383 0.04815975 0.06012258 0.2885774 0.46493632]

print('----------------------------------------------')
# m14_FI_dropfeature_masterpackage.py에서 argmax이용해서 일정 수치 이하의 feature_drop하는거랑 같은 작동원리이다.

for i,thresh in enumerate(aaa,start=1):
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)   # threshold는 그 수치 이상의 값만 가져간다. 그 이하의 features는 다 날린다.
    
    select_x_train = selection.transform(x_train)  
    select_x_test = selection.transform(x_test)  
    print(i,'회차',select_x_train.shape,select_x_test.shape)
    
    selection_model = XGBRegressor(tree_method = 'gpu_hist',predictor = 'gpu_predictor')
    selection_model.fit(select_x_train,y_train)
    
    print(f'model.score : {np.round(selection_model.score(select_x_test,y_test),4)}')
    
    selection_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test,selection_y_predict)
    
    # th = round(thresh,4)
    print(f'Thresh={str(np.round(thresh,4))}, n={select_x_train.shape[1]}, R2: {np.round(score*100,2)}\n')
    # print("Thresh=%.3f, n=%d, R2: %.2f%%\n"%(thresh,select_x_train.shape[1], score*100))
    

# index_max_acc = r2_list.index(max(r2_list))
# drop_list = np.where(model.feature_importances_ < th_list[index_max_acc])
# print(drop_list)
# x,y = load_boston(return_X_y=True)
# x = np.delete(x,drop_list,axis=1)
