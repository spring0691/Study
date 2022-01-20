import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA   # decomposition 분해
from sklearn.model_selection import train_test_split

#1. 데이터로드 및 정제


datasets = load_breast_cancer()      
y = datasets.target

for i in range(30):
    x = datasets.data  
    pca = PCA(n_components=i+1)              
    x = pca.fit_transform(x)    

    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=66, shuffle=True)

    #2. 모델
    from xgboost import XGBRegressor,XGBClassifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='error')
    # model = XGBRegressor

    #3. 훈련
    model.fit(x_train,y_train)

    #4. 평가, 예측
    results = model.score(x_test,y_test)
    print(f'칼럼개수{i+1}일때 결과 : ', results)


