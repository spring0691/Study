# 4개의 그래프가 한 화면에 나오게 만들어보기
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.model_selection import train_test_split
import warnings, numpy as np, pandas as pd,sys
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score,r2_score,f1_score
import matplotlib.pyplot as plt


def plot_feature_importances_dataset(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    # plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)


#0. 출력 관련 옵션들
warnings.filterwarnings(action='ignore')
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',50)
pd.set_option('display.max_columns',50)

#1. 데이터 로드

path = '../Project/Kaggle_Project/bike/'
Bikedata = pd.read_csv(path + 'train.csv')                 
dd =  {'Breast_cancer':load_breast_cancer(),'Iirs':load_iris(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Bike':Bikedata,'Fetch_covtype':fetch_covtype()}
#
#2. 모델링 설정
cla_model_list = [DecisionTreeClassifier(max_depth=5,random_state=66),RandomForestClassifier(max_depth=5,random_state=66),
                  GradientBoostingClassifier(random_state=66),XGBClassifier(random_state=66,eval_metric='error')]   # objective='multi:softprob' objective='binary:logistic', 
reg_model_list = [DecisionTreeRegressor(max_depth=5,random_state=66),RandomForestRegressor(max_depth=5,random_state=66),
                  GradientBoostingRegressor(random_state=66),XGBRegressor(random_state=66)]

#3. 컴파일 훈련

for name,data in dd.items():
    
    if name == 'Bike':
        x = Bikedata.drop(['casual','registered','count'], axis=1)  
        x['datetime'] = pd.to_datetime(x['datetime'])
        x['year'] = x['datetime'].dt.year
        x['month'] = x['datetime'].dt.month
        x['day'] = x['datetime'].dt.day
        x['hour'] = x['datetime'].dt.hour
        x = x.drop('datetime', axis=1)
        y = Bikedata['count']  # np.unique(y, return_counts = True 누가봐도 회귀모델
        y = np.log1p(y)
    else:
        datasets = data
        x = datasets.data
        x = pd.DataFrame(x, columns=datasets['feature_names'])
        y = datasets.target
        #print(datasets.feature_names) # feature의 이름들 확인 가능.
        
    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
    choice = np.unique(y, return_counts=True)[1].min()    
    
    print(f'{name} 데이터셋의 결과를 소개합니다~')
    
    if choice > 4:       
        print('나는 분류모델!')                                             
        
        plt.figure(figsize=(20,20))
        for i,model in enumerate(cla_model_list,start=1):
            model.fit(x_train,y_train)
            print(f'{model}.score : ', model.score(x_test,y_test))   
            print(model.feature_importances_)
            plt.subplot(2,2,i)
            plot_feature_importances_dataset(model)
        plt.show()
        
        print('\n')
        
    else:                    
        print('나는 회귀모델!')  
                                                   
        plt.figure(figsize=(20,20))
        for i,model in enumerate(reg_model_list,start=1):
            model.fit(x_train,y_train)
            print(f'{model}.score : ', model.score(x_test,y_test))   
            print(model.feature_importances_)
            plt.subplot(2,2,i)
            plot_feature_importances_dataset(model)
        plt.show()
        
        print('\n')
    
                                


'''
min_index_val = np.min(feature_importances_)
min_index = np.where(feature_importances_ == min_index_val)[0][0]
print("min_index_val ",min_index_val)
print("min_index ",min_index)

print("< min_index_val ",min_index_val)
print("< min_index ",min_index)
print(dataset.feature_names[min_index]) 
x = np.delete(x,min_index,axis=1)

feature_names = dataset.feature_names.remove(dataset.feature_names[min_index])
'''