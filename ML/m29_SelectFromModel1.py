import numpy as np, pandas as pd
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.feature_selection  import SelectFromModel

#1. 데이터

x, y = load_diabetes(return_X_y=True)   # (442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)  

scaler = StandardScaler()
real_x_train = scaler.fit_transform(x_train)
real_x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(tree_method = 'gpu_hist',predictor = 'gpu_predictor')

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
score = model.score(x_test,y_test)
print(f'model.score : {np.round(score,4)}')

# print(f'Fi : {model.feature_importances_}')
print(f'Fi_sort : {np.sort(model.feature_importances_)}')
aaa = np.sort(model.feature_importances_)#
# [0.02593723 0.03284873 0.03821951 0.04788677 0.05547738 0.06321331 0.06597804 0.07382324 0.19681737 0.39979842]

print('----------------------------------------------')
# m14_FI_dropfeature_masterpackage.py에서 argmax이용해서 일정 수치 이하의 feature_drop하는거랑 같은 작동원리이다.

# SelectFromModel은 threshold 값 지정해주면 모델을 한번 다시 돌려서 feature_inportance를 뽑은 후 그 이하의 수치를 날리고
# 칼럼정리좀 해주는? 딱 그 정도 용도이다. 제일 좋은 최적의 값을 찾으려면 for문 돌려서 조건문으로 최적 score빼내야한다.

for i,thresh in enumerate(aaa,start=1):
    selection = SelectFromModel(model, threshold=thresh, prefit=True)   # threshold는 그 수치 이상의 값만 가져간다. 그 이하의 features는 다 날린다.
    
    select_x_train = selection.transform(real_x_train)  
    select_x_test = selection.transform(real_x_test)  
    print(i,'회차',select_x_train.shape,select_x_test.shape)
    
    selection_model = XGBRegressor(tree_method = 'gpu_hist',predictor = 'gpu_predictor')
    selection_model.fit(select_x_train,y_train)
    
    print(f'model.score : {selection_model.score(select_x_test,y_test)}')
    
    selection_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test,selection_y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%\n"%(thresh,select_x_train.shape[1], score*100))
   

# y_predict = model.predict(x_test)
# print(f'r2_score : {np.round(r2_score(y_test,y_predict),4)}')