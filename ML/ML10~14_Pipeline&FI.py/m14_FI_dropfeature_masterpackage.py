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

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',100)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 170)

def plot_feature_importances_dataset1(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel(f"{str(model).split('(')[0]}")
    plt.ylabel("Features Importances")
    plt.ylim(-1,n_features)

def plot_feature_importances_dataset2(model):
    n_features = xx.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), xx.columns)
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
dd =  {'Breast_cancer':load_breast_cancer(),'Iirs':load_iris(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Bike':Bikedata,'Fetch_covtype':fetch_covtype()}  

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
    
    print(f'{name} 데이터셋의 결과를 소개합니다~\n')
    
    plt.figure(figsize=(20,20))
    plt.suptitle(name, fontsize=30)
    
    if choice > 4:       
        print('나는 분류모델!\n')                                             
        
        for i,model in enumerate(cla_model_list,start=1):
               
            model.fit(x_train,y_train)
            print('feature 제거 전')
            print(f'{str(model).split("(")[0]}.score : {model.score(x_test,y_test)}')   
            # print(pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=x.columns))
            # Feature_importances를 dataframe으로 바꿔준 후 칼럼명 넣어주고 내림차순 정렬후 누적합 리스트로 만들어줌
            Fi = pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=x.columns).sort_values(by=0,axis=1).cumsum(axis=1)         
            del_num = np.argmax(Fi > 0.25)     # argmax이용하여 몇번째 칼럼에서 특정값 초과하는지 확인 -> 여기서 나오는 개수만큼 날려주면 된다.
            del_list = Fi.columns[:del_num]    # 전체 컬럼명의 앞에서 del_num의 개수까지 담아오고 이 칼럼들을 모두 날려주면 된다.
            
            xx = x.drop(del_list,axis=1)
            xx_train,xx_test = train_test_split(xx,train_size=0.8,shuffle=True, random_state=66)
            model.fit(xx_train,y_train)
            print('feature 제거 후')
            print(f'{str(model).split("(")[0]}.score : {model.score(xx_test,y_test)}')
            print(f'{pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=xx.columns)}\n')
            plt.subplot(2,2,i)
            plot_feature_importances_dataset2(model)
        plt.show()      
    else:                    
        print('나는 회귀모델!\n')  
                                                   
        for i,model in enumerate(reg_model_list,start=1):
               
            model.fit(x_train,y_train)
            print('feature 제거 전')
            print(f'{str(model).split("(")[0]}.score : {model.score(x_test,y_test)}')   
            # print(pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=x.columns))
            # Feature_importances를 dataframe으로 바꿔준 후 칼럼명 넣어주고 내림차순 정렬후 누적합 리스트로 만들어줌
            Fi = pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=x.columns).sort_values(by=0,axis=1).cumsum(axis=1)            
            del_num = np.argmax(Fi > 0.25)     # argmax이용하여 몇번째 칼럼에서 특정값 초과하는지 확인 -> 여기서 나오는 개수만큼 날려주면 된다.
            del_list = Fi.columns[:del_num]    # 전체 컬럼명의 앞에서 del_num의 개수까지 담아오고 이 칼럼들을 모두 날려주면 된다.
            
            
            xx = x.drop(del_list,axis=1)
            xx_train,xx_test = train_test_split(xx,train_size=0.8,shuffle=True, random_state=66)
            model.fit(xx_train,y_train)
            print('feature 제거 후')
            print(f'{str(model).split("(")[0]}.score : {model.score(xx_test,y_test)}')
            print(f'{pd.DataFrame(model.feature_importances_.reshape(1,-1), columns=xx.columns)}\n')
            plt.subplot(2,2,i)
            plot_feature_importances_dataset2(model)
        plt.show()   
         
        

'''
Breast_cancer 데이터셋의 결과를 소개합니다~
나는 분류모델!

feature 제거 전
DecisionTreeClassifier.score : 0.9035087719298246
feature 제거 후
DecisionTreeClassifier.score : 0.9385964912280702
   worst area  worst concave points
0    0.844422              0.155578

feature 제거 전
RandomForestClassifier.score : 0.9649122807017544
feature 제거 후
RandomForestClassifier.score : 0.9649122807017544
   mean radius  mean area  mean concavity  mean concave points  worst radius  worst perimeter  worst area  worst concave points
0      0.05052   0.040264        0.048183             0.125913      0.164673         0.207231    0.183675              0.179541

feature 제거 전
GradientBoostingClassifier.score : 0.956140350877193
feature 제거 후
GradientBoostingClassifier.score : 0.9649122807017544
   mean concave points  worst radius  worst area  worst concave points
0             0.116107      0.294604    0.381784              0.207505

feature 제거 전
XGBClassifier.score : 0.9736842105263158
feature 제거 후
XGBClassifier.score : 0.956140350877193
   mean compactness  mean concave points  worst perimeter  worst area  worst concave points
0          0.040714             0.187467         0.056218    0.538465              0.177136



Iirs 데이터셋의 결과를 소개합니다~

나는 분류모델!

feature 제거 전
DecisionTreeClassifier.score : 0.9666666666666667
feature 제거 후
DecisionTreeClassifier.score : 0.9333333333333333
   petal length (cm)  petal width (cm)
0           0.545174          0.454826

feature 제거 전
RandomForestClassifier.score : 0.9666666666666667
feature 제거 후
RandomForestClassifier.score : 0.9666666666666667
   petal length (cm)  petal width (cm)
0           0.519975          0.480025

feature 제거 전
GradientBoostingClassifier.score : 0.9666666666666667
feature 제거 후
GradientBoostingClassifier.score : 0.9666666666666667
   petal length (cm)  petal width (cm)
0           0.276106          0.723894

feature 제거 전
XGBClassifier.score : 0.9
feature 제거 후
XGBClassifier.score : 0.9666666666666667
   petal length (cm)  petal width (cm)
0           0.510896          0.489104



Wine 데이터셋의 결과를 소개합니다~

나는 분류모델!

feature 제거 전
DecisionTreeClassifier.score : 0.9444444444444444
feature 제거 후
DecisionTreeClassifier.score : 0.9444444444444444
   flavanoids  od280/od315_of_diluted_wines   proline
0    0.201848                      0.409298  0.388854

feature 제거 전
RandomForestClassifier.score : 1.0
feature 제거 후
RandomForestClassifier.score : 1.0
   alcohol  flavanoids  color_intensity       hue  od280/od315_of_diluted_wines   proline
0  0.13783    0.191771         0.202757  0.099701                      0.152935  0.215006

feature 제거 전
GradientBoostingClassifier.score : 0.9722222222222222
feature 제거 후
GradientBoostingClassifier.score : 0.9722222222222222
   color_intensity  od280/od315_of_diluted_wines   proline
0         0.332586                      0.349312  0.318102

feature 제거 전
XGBClassifier.score : 1.0
feature 제거 후
XGBClassifier.score : 0.9722222222222222
   flavanoids  color_intensity  od280/od315_of_diluted_wines   proline
0    0.226471         0.191747                      0.381011  0.200771



Boston 데이터셋의 결과를 소개합니다~

나는 회귀모델!

feature 제거 전
DecisionTreeRegressor.score : 0.8507309980875365
feature 제거 후
DecisionTreeRegressor.score : 0.7684642446999271
         RM     LSTAT
0  0.344514  0.655486

feature 제거 전
RandomForestRegressor.score : 0.9164045544049574
feature 제거 후
RandomForestRegressor.score : 0.8064866951522273
         RM     LSTAT
0  0.469637  0.530363

feature 제거 전
GradientBoostingRegressor.score : 0.9461026657864682
feature 제거 후
GradientBoostingRegressor.score : 0.8171687035857739
         RM     LSTAT
0  0.380444  0.619556

feature 제거 전
XGBRegressor.score : 0.9221188601856797
feature 제거 후
XGBRegressor.score : 0.9251745847026621
        NOX        RM     LSTAT
0  0.117719  0.336736  0.545545



Diabets 데이터셋의 결과를 소개합니다~

나는 회귀모델!

feature 제거 전
DecisionTreeRegressor.score : 0.18699053453135217
feature 제거 후
DecisionTreeRegressor.score : 0.31881789174720987
        bmi        bp        s5
0  0.297694  0.147001  0.555304

feature 제거 전
RandomForestRegressor.score : 0.3956821110335006
feature 제거 후
RandomForestRegressor.score : 0.39080312783601767
        bmi        bp        s5
0  0.418098  0.172506  0.409396

feature 제거 전
GradientBoostingRegressor.score : 0.3919166774198126
feature 제거 후
GradientBoostingRegressor.score : 0.33704409324701723
        age       bmi       bp        s5
0  0.098483  0.337845  0.15002  0.413651

feature 제거 전
XGBRegressor.score : 0.23802704693460175
feature 제거 후
XGBRegressor.score : 0.2758251357523339
        bmi        bp        s3        s5        s6
0  0.237984  0.128225  0.103098  0.427969  0.102725



Bike 데이터셋의 결과를 소개합니다~

나는 회귀모델!

feature 제거 전
DecisionTreeRegressor.score : 0.8242695706277716
feature 제거 후
DecisionTreeRegressor.score : 0.7142695229222356
   hour
0   1.0

feature 제거 전
RandomForestRegressor.score : 0.8382811515116764
feature 제거 후
RandomForestRegressor.score : 0.7160941253682609
   hour
0   1.0

feature 제거 전
GradientBoostingRegressor.score : 0.920359648360825
feature 제거 후
GradientBoostingRegressor.score : 0.7189282562382173
   hour
0   1.0

feature 제거 전
XGBRegressor.score : 0.951330931187238
feature 제거 후
XGBRegressor.score : 0.8419952713803145
   workingday      year      hour
0    0.136339  0.072735  0.790926


Fetch_covtype 데이터셋의 결과를 소개합니다~

나는 분류모델!

feature 제거 전
DecisionTreeClassifier.score : 0.7028734197912274
feature 제거 후
DecisionTreeClassifier.score : 0.6711874908565183
   Elevation
0        1.0

feature 제거 전
RandomForestClassifier.score : 0.676514375704586
feature 제거 후
RandomForestClassifier.score : 0.6815142466201389
   Elevation  Horizontal_Distance_To_Roadways  Wilderness_Area_3  Soil_Type_3  Soil_Type_9  Soil_Type_11  Soil_Type_21  Soil_Type_37
0   0.543945                         0.049035            0.19647     0.042931     0.043775      0.043796      0.049808      0.030239

feature 제거 전
GradientBoostingClassifier.score : 0.773491217954786
feature 제거 후
GradientBoostingClassifier.score : 0.7300758155985646
   Elevation  Horizontal_Distance_To_Hydrology  Horizontal_Distance_To_Roadways  Horizontal_Distance_To_Fire_Points
0    0.77732                          0.068973                         0.093577                             0.06013

feature 제거 전
XGBClassifier.score : 0.869392356479609
feature 제거 후
XGBClassifier.score : 0.7264442398216914
   Elevation  Wilderness_Area_0  Wilderness_Area_1  Wilderness_Area_2  Wilderness_Area_3  Soil_Type_1  Soil_Type_2  Soil_Type_3  Soil_Type_9  Soil_Type_11  Soil_Type_20  Soil_Type_21  \
0   0.069987            0.06927           0.026342           0.030161            0.05328     0.058381     0.023506     0.059828     0.020145      0.091477      0.040281      0.060697

   Soil_Type_22  Soil_Type_23  Soil_Type_26  Soil_Type_28  Soil_Type_29  Soil_Type_30  Soil_Type_31  Soil_Type_32  Soil_Type_34  Soil_Type_37  Soil_Type_38
0      0.045951      0.023281      0.023883      0.035048      0.034411      0.026914      0.054939      0.021917      0.023163      0.048801      0.058335
'''