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
    plt.xlabel(f"{str(model).split('(')[0]}")
    plt.ylabel("Features Importances")
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
    
    
    
    plt.figure(figsize=(20,20))
    plt.suptitle(name, fontsize=30)
    if choice > 4:       
        print('나는 분류모델!\n')                                             
        
        for i,model in enumerate(cla_model_list,start=1):
            model.fit(x_train,y_train)
            print(f'{str(model).split("(")[0]}.score : {model.score(x_test,y_test)}')   
            print(f'{model.feature_importances_}\n')
            plt.subplot(2,2,i)
            plot_feature_importances_dataset(model)
        plt.show()
        
        print('\n')
        
    else:                    
        print('나는 회귀모델!\n')  
                                                   
        for i,model in enumerate(reg_model_list,start=1):
            model.fit(x_train,y_train)
            print(f'{str(model).split("(")[0]}.score : {model.score(x_test,y_test)}')   
            print(f'{model.feature_importances_}\n')
            plt.subplot(2,2,i)
            plot_feature_importances_dataset(model)
        plt.show()
        
        print('\n')
    
'''
Breast_cancer 데이터셋의 결과를 소개합니다~
나는 분류모델!

DecisionTreeClassifier.score : 0.9035087719298246
[0.         0.06054151 0.         0.         0.         0.
 0.         0.02005078 0.         0.02291518 0.         0.
 0.         0.01973513 0.         0.         0.00636533 0.00442037
 0.         0.004774   0.         0.01642816 0.         0.72839202
 0.         0.         0.00470676 0.11167078 0.         0.        ]

RandomForestClassifier.score : 0.9649122807017544
[0.03562902 0.0164037  0.03051708 0.03457756 0.00433399 0.00471353
 0.06756296 0.10437045 0.00582812 0.00432088 0.01141483 0.00449143
 0.01214206 0.02714175 0.00393645 0.00336399 0.0032444  0.00211765
 0.00313246 0.00476762 0.14081098 0.01349577 0.13041543 0.15821168
 0.00871971 0.01521997 0.03190766 0.1007862  0.00816395 0.00825876]

GradientBoostingClassifier.score : 0.956140350877193
[7.20294708e-05 3.62284081e-02 5.95384448e-04 3.41616186e-03
 4.64512832e-06 4.36397371e-03 3.55435800e-04 1.20138760e-01
 8.61929292e-04 3.91191305e-04 4.01660879e-03 4.53620568e-06
 4.86011664e-04 1.80765655e-02 3.89557796e-04 5.58761824e-05
 3.81499976e-03 1.54718982e-03 3.39325936e-05 1.09672550e-03
 3.33241151e-01 4.20711663e-02 4.25126125e-02 2.60162667e-01
 6.52931144e-03 1.17294722e-04 1.40567435e-02 1.03355815e-01
 4.11779309e-06 1.99919769e-03]

XGBClassifier.score : 0.9736842105263158
[0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
 0.0054994  0.09745206 0.00340272 0.00369179 0.00769183 0.00281184
 0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
 0.00639412 0.0050556  0.01813928 0.02285904 0.22248559 0.2849308
 0.00233393 0.         0.00903706 0.11586287 0.00278498 0.00775311]



Iirs 데이터셋의 결과를 소개합니다~
나는 분류모델!

DecisionTreeClassifier.score : 0.9666666666666667
[0.         0.0125026  0.53835801 0.44913938]

RandomForestClassifier.score : 0.9666666666666667
[0.08150824 0.02190985 0.46987909 0.42670282]

GradientBoostingClassifier.score : 0.9666666666666667
[0.00225937 0.01496914 0.39208442 0.59068707]

XGBClassifier.score : 0.9
[0.01835513 0.0256969  0.6204526  0.33549538]



Wine 데이터셋의 결과를 소개합니다~
나는 분류모델!

DecisionTreeClassifier.score : 0.9444444444444444
[0.01598859 0.00489447 0.         0.         0.         0.
 0.1569445  0.         0.         0.04078249 0.08604186 0.33215293
 0.36319516]

RandomForestClassifier.score : 1.0
[0.14074602 0.02731994 0.01584483 0.04709609 0.02201035 0.06085119
 0.1769038  0.01467615 0.02739868 0.12602185 0.07531074 0.1187375
 0.14708285]

GradientBoostingClassifier.score : 0.9722222222222222
[1.53477029e-02 4.21296150e-02 2.40281943e-02 3.35194115e-03
 2.50709383e-03 3.48521584e-05 1.06037687e-01 1.26209560e-04
 1.66667944e-04 2.50956056e-01 2.98140736e-02 2.48782846e-01
 2.76717060e-01]

XGBClassifier.score : 1.0
[0.01854127 0.04139537 0.01352911 0.01686821 0.02422602 0.00758254
 0.10707159 0.01631111 0.00051476 0.12775213 0.01918284 0.50344414
 0.10358089]



Boston 데이터셋의 결과를 소개합니다~
나는 회귀모델!

DecisionTreeRegressor.score : 0.8507309980875365
[0.0372032  0.         0.         0.         0.01464736 0.29092518
 0.         0.05968885 0.         0.00583002 0.         0.01786395
 0.57384145]

RandomForestRegressor.score : 0.9164045544049574
[0.03546375 0.00071493 0.00293147 0.00074901 0.02303277 0.40451793
 0.00750937 0.0595624  0.00212047 0.01116823 0.01525436 0.00494847
 0.43202684]

GradientBoostingRegressor.score : 0.9461026657864682
[2.43273595e-02 2.11578647e-04 2.24430444e-03 2.38614237e-04
 4.10279821e-02 3.57836966e-01 5.97748092e-03 8.45867788e-02
 2.47259505e-03 1.11046127e-02 3.37005090e-02 6.43038205e-03
 4.29840837e-01]

XGBRegressor.score : 0.9221188601856797
[0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
 0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
 0.4284835 ]



Diabets 데이터셋의 결과를 소개합니다~
나는 회귀모델!

DecisionTreeRegressor.score : 0.18699053453135217
[0.04446378 0.         0.24919201 0.11505227 0.         0.04366568
 0.03928846 0.         0.45459058 0.05374722]

RandomForestRegressor.score : 0.3956821110335006
[0.04935835 0.00572708 0.33488576 0.1119164  0.02336201 0.04024662
 0.02660828 0.0138166  0.3307613  0.0633176 ]

GradientBoostingRegressor.score : 0.3919166774198126
[0.0595394  0.01148448 0.27560272 0.11832653 0.02361885 0.05343679
 0.04069013 0.01671989 0.34304014 0.05754107]

XGBRegressor.score : 0.23802704693460175
[0.02593721 0.03821949 0.19681741 0.06321319 0.04788679 0.05547739
 0.07382318 0.03284872 0.39979857 0.06597802]



Bike 데이터셋의 결과를 소개합니다~
나는 회귀모델!

DecisionTreeRegressor.score : 0.8242695706277716
[0.         0.         0.04645059 0.00102855 0.04600898 0.
 0.00479315 0.         0.03452569 0.02321523 0.         0.84397781]

RandomForestRegressor.score : 0.8382811515116764
[7.44221167e-03 1.44258670e-05 4.72026827e-02 2.38192255e-03
 4.43328834e-02 3.68364486e-03 6.72840197e-03 1.57610613e-04
 3.18079880e-02 2.12617999e-02 1.42531675e-04 8.34843897e-01]

GradientBoostingRegressor.score : 0.920359648360825
[9.43835859e-03 2.31659346e-04 5.21851127e-02 1.11892748e-02
 3.52842294e-02 1.32219554e-02 1.21663385e-02 1.13926300e-03
 3.51429903e-02 3.16501014e-02 7.33946687e-04 7.97616770e-01]

XGBRegressor.score : 0.951330931187238
[0.06115152 0.01534587 0.18584788 0.02160099 0.04729559 0.01876757
 0.01368091 0.0048362  0.13472676 0.0443396  0.0061127  0.44629446]
 
 
Fetch_covtype 데이터셋의 결과를 소개합니다~
나는 분류모델!

DecisionTreeClassifier.score : 0.7028734197912274
[0.82214481 0.00178621 0.         0.02599236 0.0031613  0.01463683
 0.         0.02456075 0.         0.00264821 0.00819376 0.
 0.015303   0.         0.         0.02888085 0.         0.032297
 0.         0.         0.         0.         0.         0.0036305
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.01676442 0.         0.
 0.         0.         0.         0.         0.         0.        ]

RandomForestClassifier.score : 0.676514375704586
[3.75338908e-01 6.34992388e-03 8.80997860e-03 6.02170573e-03
 4.41792838e-03 3.22100212e-02 5.10967309e-03 7.08894244e-03
 2.83781532e-03 2.08471079e-02 2.67870284e-02 7.60178573e-03
 1.87448388e-02 1.53262951e-01 2.34321998e-03 2.19097564e-02
 2.41560881e-03 3.00927561e-02 2.73892869e-04 8.10403524e-03
 1.06051710e-06 1.02444774e-06 3.28853202e-05 5.23106604e-02
 1.45840932e-03 4.45236414e-02 4.59327384e-03 9.27309479e-05
 1.01837660e-06 1.56350754e-05 2.63058491e-04 2.68724923e-04
 3.85620109e-05 1.35369227e-04 8.05113522e-05 4.56274529e-02
 2.46591315e-02 2.81222307e-04 2.31468828e-05 2.25224979e-04
 0.00000000e+00 0.00000000e+00 7.41869965e-03 1.44316575e-03
 2.20578197e-04 1.75890208e-03 1.29220818e-03 5.01564677e-06
 7.26722912e-04 0.00000000e+00 2.07416774e-04 3.43637281e-02
 2.62477722e-02 1.11151687e-02]

GradientBoostingClassifier.score : 0.773491217954786
[6.47964052e-01 7.01979696e-03 1.25933225e-03 3.83555246e-02
 9.43188004e-03 5.68955566e-02 7.15281804e-03 2.54020054e-02
 2.76021972e-03 4.16122345e-02 2.53260300e-02 4.98964512e-03
 1.33829696e-02 2.22515945e-03 3.01423557e-04 1.29983880e-02
 4.43319333e-03 1.69606834e-02 9.36700224e-04 1.55418805e-03
 0.00000000e+00 0.00000000e+00 6.43917258e-05 2.37365650e-03
 1.18185981e-03 6.25397649e-03 1.40788404e-03 3.41096450e-04
 5.80363416e-06 4.90335609e-04 1.35560962e-03 5.01191346e-05
 3.26666517e-04 1.44955041e-03 1.03001875e-03 1.65777214e-02
 1.15901253e-02 1.18624297e-03 5.81281617e-05 5.38181436e-05
 1.16634895e-03 7.23694225e-05 3.84602811e-03 1.27728754e-03
 3.89661356e-03 8.54018587e-03 8.21391884e-04 7.80996812e-04
 1.61048489e-03 3.95975636e-05 8.54865295e-04 2.98320116e-03
 6.16999630e-03 1.18182694e-03]

XGBClassifier.score : 0.869392356479609
[0.08789405 0.00749023 0.00467787 0.01391024 0.00717675 0.0134737
 0.00988278 0.01234562 0.00532679 0.01228409 0.05897678 0.02524585
 0.03400296 0.02194732 0.0036819  0.04537919 0.02427015 0.03912758
 0.00495591 0.00572592 0.00168255 0.0079334  0.00965479 0.01421824
 0.01182582 0.04753365 0.00966461 0.00257953 0.0009051  0.00779884
 0.01206601 0.00594357 0.00512652 0.01335775 0.02019449 0.04943318
 0.02908161 0.01685023 0.00888427 0.00683316 0.01511435 0.00262304
 0.02676792 0.01816068 0.02461869 0.04864944 0.01937804 0.0056233
 0.01617088 0.00294164 0.00907579 0.03714141 0.0415679  0.01282378]
'''              


