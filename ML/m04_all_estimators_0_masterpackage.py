from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes()} # ,fetch_covtype()
scaler = MinMaxScaler()
classifier_all = all_estimators(type_filter='classifier')  
regressor_all = all_estimators(type_filter='regressor')

for name,data in dd.items():
    datasets = data
    x = datasets.data
    y = datasets.target
    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    choice = len(np.unique(y))
    
    print(f'{name} 데이터셋의 결과를 소개합니다~\n')
    
    if choice < 10:        
        for (cn, cl) in classifier_all:
            try:
                model = cl()
                model.fit(x_train,y_train)
                y_predict = model.predict(x_test)
                acc = accuracy_score(y_test,y_predict)
                print(cn, '의 정답률 : ', acc)
            except:
                # print(cn,'에서 오류떴어~')
                pass
        print('\n\n')
        
    elif choice > 10:
        for (rn, rl) in regressor_all:               
            try:
                model = rl()   
                model.fit(x_train,y_train)
                y_predict = model.predict(x_test)
                r2 = r2_score(y_test,y_predict)
                print(rn, '의 정답률 : ', r2)
            except:
                # print(rn,'에서 오류떴어~')
                pass
        print('\n\n')

