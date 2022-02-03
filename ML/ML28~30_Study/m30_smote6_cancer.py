# label 0은 212개 1은 357개인데
# label 0을 112개 삭제한 상태로 시작.

# smote 넣고 안넣고 비교.

import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape)              # (569, 30) (569,)
# print(pd.Series(y).value_counts())  # 1    357   0    212

# 여기서 label이 0인 행 112개를 삭제해서 0인 행의 개수가 100개가 되게 해보자. 당연히 x도 세트로 삭제해줘야한다.

index_list = np.where(y==0) # y에서 0이 들어있는 인덱스 위치가 담긴 리스트
# print(len(index_list[0])) # 개수를 확인해보면 딱 212개가 맞다.

del_index_list = index_list[0][100:]
# print(len(del_index_list))    # 100개가 나왔다.

new_x = np.delete(x,del_index_list,axis=0) # del_index_list
new_y = np.delete(y,del_index_list)

x_train,x_test,y_train,y_test = train_test_split(new_x,new_y,shuffle=True, random_state=66, train_size=0.8,stratify=new_y)

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
    eval_metric='error',
    use_label_encoder=False,
)

model.fit(x_train,y_train,verbose=True,early_stopping_rounds=100,eval_set=[(x_val,y_val)])

print(f"md.score : {round(model.score(x_test,y_test),4)}")                             
print(f"ac_score : {round(accuracy_score(y_test,model.predict(x_test)),4)}")            
print(f"f1_score : {round(f1_score(y_test,model.predict(x_test),average='macro'),4)}")   
print(f"f1_score : {round(f1_score(y_test,model.predict(x_test),average='micro'),4)}") 

'''
upsampling X
md.score : 0.9674
ac_score : 0.9674
f1_score : 0.9512
f1_score : 0.9674

upsampling O
md.score : 0.9674
ac_score : 0.9674
f1_score : 0.9529       < -- 0.0017 상승했다
f1_score : 0.9674

'''