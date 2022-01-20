# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 
# 데이터셋 재구성 후 각 모델별로 돌려서 결과 도출!
# 기존 모델결과와의 비교
# DecisionTree,RandomForest,GradientBoosting,XGB 모델 4개 사용.

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
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel(f"{str(model).split('(')[0]}")
    plt.ylabel("Features Importances")
    plt.ylim(-1,n_features)


#0. 출력 관련 옵션들
warnings.filterwarnings(action='ignore')
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 190)
#1. 데이터 로드

path = '../Project/Kaggle_Project/bike/'
Bikedata = pd.read_csv(path + 'train.csv')                 
dd =  {'Breast_cancer':load_breast_cancer(),'Iirs':load_iris(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Bike':Bikedata} # ,'Fetch_covtype':fetch_covtype()
#
#2. 모델링 설정
cla_model_list = [DecisionTreeClassifier(max_depth=5,random_state=66),RandomForestClassifier(max_depth=5,random_state=66),
                  GradientBoostingClassifier(random_state=66),XGBClassifier(random_state=66,eval_metric='error')]   # objective='multi:softprob' objective='binary:logistic', 
reg_model_list = [DecisionTreeRegressor(max_depth=5,random_state=66),RandomForestRegressor(max_depth=5,random_state=66),
                  GradientBoostingRegressor(random_state=66),XGBRegressor(random_state=66)]


#3. 컴파일 훈련

# del_list를 뽑기위한 반복문
#all_del_list={}
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
        
    #x = x.drop(columns=0,axis=1)  # 이부분에서 feature 제거 가능
    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
    choice = np.unique(y, return_counts=True)[1].min()    
    
    print(f'{name} 데이터셋의 결과를 소개합니다~')
    
    plt.figure(figsize=(20,20))
    plt.suptitle(name, fontsize=30)
    
    if choice > 4:       
        print('나는 분류모델!\n')                                             
        
        for i,model in enumerate(cla_model_list,start=1):
            model.fit(x_train,y_train)
            print(f'{str(model).split("(")[0]}.score : {model.score(x_test,y_test)}')   
            #print(f'{model.feature_importances_}\n')
            # Feature_importances를 dataframe으로 바꿔준 후 칼럼명 넣어주고 내림차순 정렬후 누적합 리스트로 만들어줌
            Fi = pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=x.columns).sort_values(by=0,axis=1).cumsum(axis=1)            
            del_num = np.argmax(Fi > 0.25)     # argmax이용하여 몇번째 칼럼에서 특정값 초과하는지 확인 -> 여기서 나오는 개수만큼 날려주면 된다.
            del_list = Fi.columns[:del_num]    # 전체 컬럼명의 앞에서 del_num의 개수까지 담아오고 이 칼럼들을 모두 날려주면 된다.
            
            xx = x.drop(del_list,axis=1)
            xx_train,xx_test = train_test_split(xx,train_size=0.8,shuffle=True, random_state=66)
            model.fit(xx_train,y_train)
            print(f'{str(model).split("(")[0]}.score : {model.score(xx_test,y_test)}')
            exit()
        #     plt.subplot(2,2,i)
        #     plot_feature_importances_dataset(model)
        # plt.show()

        print('\n')
        
    else:                    
        print('나는 회귀모델!\n')  
                                                   
        for i,model in enumerate(reg_model_list,start=1):
            model.fit(x_train,y_train)
            print(f'{str(model).split("(")[0]}.score : {model.score(x_test,y_test)}')   
            #print(f'{model.feature_importances_}\n')
            # Feature_importances를 dataframe으로 바꿔준 후 칼럼명 넣어주고 내림차순 정렬후 누적합 리스트로 만들어줌
            Fi = pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=x.columns).sort_values(by=0,axis=1).cumsum(axis=1)            
            del_num = np.argmax(Fi > 0.25)     # argmax이용하여 몇번째 칼럼에서 특정값 초과하는지 확인 -> 여기서 나오는 개수만큼 날려주면 된다.
            del_list = Fi.columns[:del_num]    # 전체 컬럼명의 앞에서 del_num의 개수까지 담아오고 이 칼럼들을 모두 날려주면 된다.
            
        #     plt.subplot(2,2,i)
        #     plot_feature_importances_dataset(model)
        # plt.show()
        
        print('\n')

