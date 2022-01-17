import pandas as pd, os, numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings(action='ignore')

path = '../Project/Kaggle_Project/bike/'

train = pd.read_csv(path + 'train.csv')                 

x = train.drop(['datetime','casual','registered','count'], axis=1)  
y = train['count']  # np.unique(y, return_counts = True 누가봐도 회귀모델
 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

regressor_all = all_estimators(type_filter='regressor')

for i,(rn,rl) in enumerate(regressor_all,start=1):
    try:
        model = rl()   
        regressor_scores = cross_val_score(model,x,y,cv=kfold)
        print(f'{i}번째 {rn}의 정답률 : ', np.round(regressor_scores,4))
    except:
        print(f'{i}번째 {rn}에서 오류떴어~')
        
'''
1번째 ARDRegression의 정답률 :  [0.2493 0.2767 0.2538 0.256  0.2637]
2번째 AdaBoostRegressor의 정답률 :  [0.1599 0.2046 0.2145 0.2218 0.2421]
3번째 BaggingRegressor의 정답률 :  [0.2126 0.2193 0.2451 0.2395 0.2471]
4번째 BayesianRidge의 정답률 :  [0.2494 0.2766 0.2539 0.2557 0.2639]
5번째 CCA의 정답률 :  [-0.1875 -0.0587 -0.1567 -0.1454 -0.1056]
6번째 DecisionTreeRegressor의 정답률 :  [-0.1923 -0.1304 -0.1101 -0.1814 -0.0865]
7번째 DummyRegressor의 정답률 :  [-0.0006 -0.0004 -0.     -0.0017 -0.    ]       
8번째 ElasticNet의 정답률 :  [0.2474 0.2742 0.2538 0.2539 0.2607]
9번째 ElasticNetCV의 정답률 :  [0.2351 0.2573 0.2426 0.2404 0.2434]
10번째 ExtraTreeRegressor의 정답률 :  [-0.1629 -0.0606 -0.0866 -0.1235 -0.0393]
11번째 ExtraTreesRegressor의 정답률 :  [0.1519 0.2079 0.2015 0.1972 0.2233]
12번째 GammaRegressor의 정답률 :  [0.1779 0.1839 0.1687 0.1655 0.1848]
13번째 GaussianProcessRegressor의 정답률 :  [-0.2971 -0.2228 -0.2674 -0.2929 -0.2117]
14번째 GradientBoostingRegressor의 정답률 :  [0.3252 0.3331 0.3221 0.3344 0.3356]
15번째 HistGradientBoostingRegressor의 정답률 :  [0.3443 0.3367 0.3445 0.3609 0.3584]
16번째 HuberRegressor의 정답률 :  [0.2334 0.2536 0.23   0.2195 0.2419]
17번째 IsotonicRegression의 정답률 :  [nan nan nan nan nan]
18번째 KNeighborsRegressor의 정답률 :  [0.1803 0.2201 0.1939 0.1979 0.2275]
19번째 KernelRidge의 정답률 :  [0.2304 0.2596 0.2377 0.2359 0.2532]
20번째 Lars의 정답률 :  [0.2495 0.2766 0.2538 0.2555 0.2641]
21번째 LarsCV의 정답률 :  [0.2492 0.2657 0.2503 0.2558 0.2556]
22번째 Lasso의 정답률 :  [0.2492 0.2766 0.2541 0.2559 0.2634]
23번째 LassoCV의 정답률 :  [0.2314 0.2542 0.2406 0.238  0.2411]
24번째 LassoLars의 정답률 :  [-0.0006 -0.0004 -0.     -0.0017 -0.    ]
25번째 LassoLarsCV의 정답률 :  [0.2495 0.27   0.2539 0.2555 0.264 ]
26번째 LassoLarsIC의 정답률 :  [0.2495 0.2764 0.2543 0.2555 0.2632]
27번째 LinearRegression의 정답률 :  [0.2495 0.2766 0.2538 0.2555 0.2641]
28번째 LinearSVR의 정답률 :  [0.2161 0.2364 0.2108 0.2078 0.2345]
29번째 MLPRegressor의 정답률 :  [0.2846 0.3072 0.2786 0.2796 0.2908]
30번째 MultiOutputRegressor에서 오류떴어~
31번째 MultiTaskElasticNet의 정답률 :  [nan nan nan nan nan]
32번째 MultiTaskElasticNetCV의 정답률 :  [nan nan nan nan nan]
33번째 MultiTaskLasso의 정답률 :  [nan nan nan nan nan]
34번째 MultiTaskLassoCV의 정답률 :  [nan nan nan nan nan]
35번째 NuSVR의 정답률 :  [0.218  0.2271 0.2126 0.206  0.2156]
36번째 OrthogonalMatchingPursuit의 정답률 :  [0.1394 0.1579 0.1659 0.1493 0.1621]
37번째 OrthogonalMatchingPursuitCV의 정답률 :  [0.2299 0.2555 0.2394 0.2353 0.2418]
38번째 PLSCanonical의 정답률 :  [-0.6781 -0.4908 -0.6098 -0.5755 -0.5253]
39번째 PLSRegression의 정답률 :  [0.2431 0.271  0.249  0.2496 0.2548]
40번째 PassiveAggressiveRegressor의 정답률 :  [-0.5569  0.1151 -0.5264 -0.2697 -0.1081]
41번째 PoissonRegressor의 정답률 :  [0.2633 0.2876 0.2602 0.2632 0.2748]
'''