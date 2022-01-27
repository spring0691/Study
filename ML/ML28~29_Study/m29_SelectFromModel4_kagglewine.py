# GridSearchCV 적용해서 출력한 값에 피처임포턴스 추출후
# SelectFromModel 만들어서 컬럼 축소 후 모델구축해서 결과 도출

# feature 줄여가면서 최적의 값 뽑아보자. 

from tabnanny import verbose
import numpy as np, pandas as pd, warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.metrics import f1_score,accuracy_score
from sklearn.feature_selection  import SelectFromModel

warnings.filterwarnings(action='ignore')

#1. 데이터

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)
x = datasets.drop(['quality'],axis=1)   # (4898, 11)    
y = datasets['quality']  

le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y)  
# x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66,stratify=y_train)  

# parameters = {"n_estimators":[2000],"learning_rate":[0.025,0.05,0.075,0.01],"max_depth":[3,5,7,9,11],"reg_alpha":[0,1],"reg_lambda":[0,1]}
# 1*4*5*2*2 = 80
#2. 모델
# model = RandomizedSearchCV(
#             XGBClassifier(tree_method = 'gpu_hist',predictor = 'gpu_predictor',eval_metric='merror',use_label_encoder=False),
#             parameters,cv=5,random_state=66,n_iter=80,verbose=2,n_jobs=-1)

model = XGBClassifier(
    n_estimators = 200,
    max_depth = 9,
    learning_rate = 0.05,
    reg_lambda = 1,
    reg_alpha = 1,
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor',
    eval_metric='merror',
    use_label_encoder=False,
)

#3. 훈련
model.fit(x_train,y_train,verbose=1) # ,early_stopping_rounds=100,eval_set=[(x_val,y_val)]

#4. 평가,예측
# print(f"최적의 파라미터 : {model.best_params_}\n")     
#    {'reg_lambda': 1, 'reg_alpha': 1, 'n_estimators': 200, 'max_depth': 9, 'learning_rate': 0.05}
print(f"md.score : {model.score(x_test,y_test)}")                               # 0.6571428571428571
print(f"ac_score : {accuracy_score(y_test,model.predict(x_test))}")             # 0.6571428571428571
print(f"f1_score : {f1_score(y_test,model.predict(x_test),average='macro')}")   #f1_score : 0.4084644366910151
print(model.feature_importances_)







'''
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
'''
