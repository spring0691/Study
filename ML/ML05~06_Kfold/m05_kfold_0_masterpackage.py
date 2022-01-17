from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.model_selection import train_test_split,KFold, cross_val_score,StratifiedKFold
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)             # 회귀모델에서 씀
Skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)  # 분류모델에서 씀

dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes(),'Fetch_covtype':fetch_covtype()} # 

classifier_all = all_estimators(type_filter='classifier')  
regressor_all = all_estimators(type_filter='regressor')
scaler = MinMaxScaler()


for name,data in dd.items():
    datasets = data
    x = datasets.data
    y = datasets.target
    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
  
    choice = len(np.unique(y))
    
    print(f'{name} 데이터셋의 결과를 소개합니다~\n')
    
    if choice < 10:        
        for (cn, cl) in classifier_all:
            try:
                model = cl()
                classifier_scores = cross_val_score(model,x,y,cv=Skfold)
                print(cn, '의 정답률 : ', np.round(classifier_scores,4))
            except:
                print(cn,'에서 오류떴어~')
        print('\n\n')
        
    elif choice > 10:
        for (rn, rl) in regressor_all:               
            try:
                model = rl()   
                regressor_scores = cross_val_score(model,x,y,cv=kfold)
                print(rn, '의 정답률 : ', np.round(regressor_scores,4))
            except:
                print(rn,'에서 오류떴어~')
        print('\n\n')
        
'''
 
Iirs 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  [0.9667 0.9667 1.     0.9    0.9333]
BaggingClassifier 의 정답률 :  [0.9667 0.9667 1.     0.9333 0.9333]
BernoulliNB 의 정답률 :  [0.3333 0.3333 0.3333 0.3333 0.3333]
CalibratedClassifierCV 의 정답률 :  [0.9333 0.9667 1.     0.8333 0.9   ]
CategoricalNB 의 정답률 :  [0.9    0.9667 0.9667 0.8667 0.9333]
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  [0.6667 0.6667 0.6667 0.6667 0.6667]
DecisionTreeClassifier 의 정답률 :  [0.9667 1.     1.     0.9    0.9333]
DummyClassifier 의 정답률 :  [0.3333 0.3333 0.3333 0.3333 0.3333]
ExtraTreeClassifier 의 정답률 :  [0.9333 0.9667 0.9333 0.9    0.9333]
ExtraTreesClassifier 의 정답률 :  [0.9667 0.9667 1.     0.9333 0.9   ]
GaussianNB 의 정답률 :  [0.9333 0.9667 1.     0.9333 0.9   ]
GaussianProcessClassifier 의 정답률 :  [0.9667 0.9667 1.     0.9    0.9333]
GradientBoostingClassifier 의 정답률 :  [0.9667 1.     1.     0.9    0.9333]
HistGradientBoostingClassifier 의 정답률 :  [0.9667 0.9667 1.     0.9333 0.9333]
KNeighborsClassifier 의 정답률 :  [0.9667 0.9667 1.     1.     0.9333]
LabelPropagation 의 정답률 :  [0.9667 0.9333 1.     0.9667 0.9333]
LabelSpreading 의 정답률 :  [0.9667 0.9333 1.     0.9667 0.9333]
LinearDiscriminantAnalysis 의 정답률 :  [1.  1.  1.  1.  0.9]
LinearSVC 의 정답률 :  [1.     1.     1.     0.9333 0.9   ]
LogisticRegression 의 정답률 :  [0.9667 1.     1.     0.9333 0.9333]
LogisticRegressionCV 의 정답률 :  [0.9667 1.     1.     1.     0.9333]
MLPClassifier 의 정답률 :  [1.  1.  1.  1.  0.9]
MultiOutputClassifier 에서 오류떴어~
MultinomialNB 의 정답률 :  [0.9667 0.9667 1.     0.9333 0.9   ]
NearestCentroid 의 정답률 :  [0.9    0.9333 1.     0.8333 0.9667]
NuSVC 의 정답률 :  [0.9667 0.9667 1.     0.9    0.9333]
OneVsOneClassifier 에서 오류떴어~
OneVsRestClassifier 에서 오류떴어~
OutputCodeClassifier 에서 오류떴어~
PassiveAggressiveClassifier 의 정답률 :  [0.9 0.9 0.9 0.9 0.8]
Perceptron 의 정답률 :  [0.9667 0.8    0.9667 0.7    0.8   ]
QuadraticDiscriminantAnalysis 의 정답률 :  [1.     1.     1.     0.9667 0.9   ]
RadiusNeighborsClassifier 의 정답률 :  [0.9333 0.9667 1.     0.8667 0.9667]
RandomForestClassifier 의 정답률 :  [0.9667 0.9667 1.     0.9333 0.9333]
RidgeClassifier 의 정답률 :  [0.8    0.9    0.8333 0.8    0.8   ]
RidgeClassifierCV 의 정답률 :  [0.8    0.9    0.8333 0.8    0.8   ]
SGDClassifier 의 정답률 :  [0.7    0.9333 0.7    0.8667 0.9333]
SVC 의 정답률 :  [0.9667 0.9667 1.     0.9    0.9667]
StackingClassifier 에서 오류떴어~
VotingClassifier 에서 오류떴어~



Breast_cancer 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  [0.9474 0.9386 0.9561 0.9737 0.9735]
BaggingClassifier 의 정답률 :  [0.9123 0.9649 0.9649 0.9386 0.9558]
BernoulliNB 의 정답률 :  [0.6228 0.6228 0.6316 0.6316 0.6283]
CalibratedClassifierCV 의 정답률 :  [0.9211 0.9123 0.9561 0.9298 0.9115]
CategoricalNB 의 정답률 :  [   nan    nan 0.9649    nan 0.9292]
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  [0.8947 0.8947 0.9298 0.886  0.8673]
DecisionTreeClassifier 의 정답률 :  [0.9211 0.9561 0.9123 0.9123 0.9027]
DummyClassifier 의 정답률 :  [0.6228 0.6228 0.6316 0.6316 0.6283]
ExtraTreeClassifier 의 정답률 :  [0.9386 0.9386 0.9211 0.9211 0.9735]
ExtraTreesClassifier 의 정답률 :  [0.9561 0.9825 0.9825 0.9474 0.9646]
GaussianNB 의 정답률 :  [0.9386 0.9386 0.9386 0.9211 0.9646]
GaussianProcessClassifier 의 정답률 :  [0.886  0.9123 0.8947 0.9386 0.8938]
GradientBoostingClassifier 의 정답률 :  [0.9474 0.9649 0.9737 0.9386 0.9735]
HistGradientBoostingClassifier 의 정답률 :  [0.9737 0.9561 1.     0.9211 0.9646]
KNeighborsClassifier 의 정답률 :  [0.9035 0.9211 0.9386 0.9561 0.9292]
LabelPropagation 의 정답률 :  [0.4211 0.386  0.386  0.386  0.3894]
LabelSpreading 의 정답률 :  [0.4211 0.386  0.386  0.386  0.3894]
LinearDiscriminantAnalysis 의 정답률 :  [0.9561 0.9561 0.9737 0.9649 0.9469]
LinearSVC 의 정답률 :  [0.9211 0.9211 0.6316 0.8947 0.9027]
LogisticRegression 의 정답률 :  [0.9298 0.9386 0.9649 0.9649 0.9558]
LogisticRegressionCV 의 정답률 :  [0.9649 0.9386 0.9561 0.9561 0.9469]
MLPClassifier 의 정답률 :  [0.9386 0.8947 0.9298 0.9561 0.8761]
MultiOutputClassifier 에서 오류떴어~
MultinomialNB 의 정답률 :  [0.8947 0.8947 0.9298 0.886  0.8673]
NearestCentroid 의 정답률 :  [0.8596 0.8947 0.9298 0.8596 0.9027]
NuSVC 의 정답률 :  [0.8596 0.8684 0.9298 0.8421 0.885 ]
OneVsOneClassifier 에서 오류떴어~
OneVsRestClassifier 에서 오류떴어~
OutputCodeClassifier 에서 오류떴어~
PassiveAggressiveClassifier 의 정답률 :  [0.9298 0.8947 0.9298 0.8772 0.9027]
Perceptron 의 정답률 :  [0.8947 0.8772 0.7281 0.7018 0.8938]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.9737 0.9737 0.9474 0.9474 0.9381]
RadiusNeighborsClassifier 의 정답률 :  [nan nan nan nan nan]
RandomForestClassifier 의 정답률 :  [0.9474 0.9737 0.9825 0.9474 0.9646]
RidgeClassifier 의 정답률 :  [0.9561 0.9561 0.9737 0.9474 0.9469]
RidgeClassifierCV 의 정답률 :  [0.9649 0.9649 0.9825 0.9474 0.9381]
SGDClassifier 의 정답률 :  [0.7807 0.6491 0.9386 0.8684 0.7522]
SVC 의 정답률 :  [0.8772 0.9298 0.9561 0.9123 0.9115]
StackingClassifier 에서 오류떴어~
VotingClassifier 에서 오류떴어~



Wine 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  [0.9167 0.8056 0.9167 0.8857 0.5714]
BaggingClassifier 의 정답률 :  [0.9167 0.9444 1.     0.9714 0.9714]
BernoulliNB 의 정답률 :  [0.3889 0.3889 0.3889 0.4    0.4286]
CalibratedClassifierCV 의 정답률 :  [0.8889 1.     0.8889 0.9143 0.9429]
CategoricalNB 의 정답률 :  [   nan    nan 0.9167    nan    nan]
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  [0.4722 0.6667 0.7222 0.6857 0.7714]
DecisionTreeClassifier 의 정답률 :  [0.8333 0.9444 0.8611 1.     0.9714]
DummyClassifier 의 정답률 :  [0.3889 0.3889 0.3889 0.4    0.4286]
ExtraTreeClassifier 의 정답률 :  [0.9167 0.8611 0.8056 0.9714 0.8571]
ExtraTreesClassifier 의 정답률 :  [1.     0.9444 1.     1.     0.9714]
GaussianNB 의 정답률 :  [0.9722 0.9444 1.     0.9714 0.9429]
GaussianProcessClassifier 의 정답률 :  [0.5278 0.3889 0.5    0.5429 0.4571]
GradientBoostingClassifier 의 정답률 :  [0.9444 0.9722 0.9444 1.     0.9143]
HistGradientBoostingClassifier 의 정답률 :  [0.9722 0.9444 0.9444 1.     0.9714]
KNeighborsClassifier 의 정답률 :  [0.5556 0.6111 0.6944 0.7429 0.8   ]
LabelPropagation 의 정답률 :  [0.4444 0.5    0.5556 0.5429 0.4571]
LabelSpreading 의 정답률 :  [0.4444 0.5    0.5556 0.5429 0.4571]
LinearDiscriminantAnalysis 의 정답률 :  [0.9722 1.     0.9722 1.     1.    ]
LinearSVC 의 정답률 :  [0.7222 0.9444 0.9167 0.8857 0.8571]
LogisticRegression 의 정답률 :  [0.9167 0.9722 0.9444 0.9714 0.9429]
LogisticRegressionCV 의 정답률 :  [0.9444 0.9722 0.9722 0.9714 0.9714]
MLPClassifier 의 정답률 :  [0.8611 0.0278 0.5278 0.9143 0.8857]  
MultiOutputClassifier 에서 오류떴어~
MultinomialNB 의 정답률 :  [0.7222 0.9167 0.8333 0.8857 0.9143]  
NearestCentroid 의 정답률 :  [0.6111 0.7778 0.75   0.7714 0.7429]
NuSVC 의 정답률 :  [0.8333 0.9167 0.8611 0.9714 0.8571]
OneVsOneClassifier 에서 오류떴어~
OneVsRestClassifier 에서 오류떴어~
OutputCodeClassifier 에서 오류떴어~
PassiveAggressiveClassifier 의 정답률 :  [0.5    0.3333 0.5278 0.4    0.6571]
Perceptron 의 정답률 :  [0.4444 0.6667 0.6667 0.5714 0.4571]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.9722 1.     1.     1.     1.    ]
RadiusNeighborsClassifier 의 정답률 :  [nan nan nan nan nan]
RandomForestClassifier 의 정답률 :  [1.     0.9444 1.     1.     0.9714]
RidgeClassifier 의 정답률 :  [0.9722 1.     1.     0.9714 1.    ]       
RidgeClassifierCV 의 정답률 :  [0.9722 1.     0.9722 0.9714 1.    ]
SGDClassifier 의 정답률 :  [0.4167 0.7222 0.6389 0.6571 0.6571]    
SVC 의 정답률 :  [0.5556 0.7778 0.7222 0.7143 0.6857]
StackingClassifier 에서 오류떴어~
VotingClassifier 에서 오류떴어~



Boston 데이터셋의 결과를 소개합니다~

ARDRegression 의 정답률 :  [0.8013 0.7632 0.5681 0.64   0.7199]
AdaBoostRegressor 의 정답률 :  [0.9028 0.7868 0.7548 0.8263 0.8815]
BaggingRegressor 의 정답률 :  [0.8842 0.8593 0.8477 0.8651 0.8608]
BayesianRidge 의 정답률 :  [0.7938 0.8112 0.5794 0.6272 0.7072]
CCA 의 정답률 :  [0.7913 0.7383 0.3942 0.5795 0.7322]
DecisionTreeRegressor 의 정답률 :  [0.768  0.6912 0.7874 0.722  0.8272]
DummyRegressor 의 정답률 :  [-0.0005 -0.0336 -0.0048 -0.0259 -0.0028]
ElasticNet 의 정답률 :  [0.7338 0.7675 0.5998 0.6062 0.6466]
ElasticNetCV 의 정답률 :  [0.7168 0.7528 0.5912 0.5929 0.6289]
ExtraTreeRegressor 의 정답률 :  [0.5765 0.4149 0.5224 0.6768 0.7621]
ExtraTreesRegressor 의 정답률 :  [0.9361 0.8479 0.7796 0.8743 0.9344]
GammaRegressor 의 정답률 :  [-0.0006 -0.0315 -0.0046 -0.0281 -0.003 ]
GaussianProcessRegressor 의 정답률 :  [-6.0731 -5.5196 -6.3348 -6.3638 -5.3516]
GradientBoostingRegressor 의 정답률 :  [0.9459 0.8368 0.826  0.8859 0.9325]
HistGradientBoostingRegressor 의 정답률 :  [0.9324 0.8242 0.7874 0.8888 0.8577]
HuberRegressor 의 정답률 :  [0.7496 0.6619 0.5285 0.4034 0.6148]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.5901 0.6811 0.5568 0.4033 0.4118]
KernelRidge 의 정답률 :  [0.8333 0.7671 0.5305 0.5836 0.7123]
Lars 의 정답률 :  [0.7747 0.7984 0.5904 0.6408 0.6844]
LarsCV 의 정답률 :  [0.8014 0.7757 0.5781 0.6007 0.7083]
Lasso 의 정답률 :  [0.7241 0.7603 0.6014 0.6046 0.6379]
LassoCV 의 정답률 :  [0.7131 0.7914 0.6073 0.6162 0.6614]
LassoLars 의 정답률 :  [-0.0005 -0.0336 -0.0048 -0.0259 -0.0028]
LassoLarsCV 의 정답률 :  [0.803  0.7757 0.5781 0.6007 0.7249]
LassoLarsIC 의 정답률 :  [0.8131 0.7977 0.5901 0.6397 0.7242]
LinearRegression 의 정답률 :  [0.8111 0.7984 0.5903 0.6408 0.7233]
LinearSVR 의 정답률 :  [0.7801 0.6603 0.5716 0.4569 0.5516]
MLPRegressor 의 정답률 :  [0.4575 0.3787 0.4331 0.4591 0.4763]
MultiOutputRegressor 에서 오류떴어~
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.2594 0.3343 0.2639 0.1191 0.1706]
OrthogonalMatchingPursuit 의 정답률 :  [0.5828 0.5659 0.4869 0.5155 0.5205]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.7526 0.7509 0.5233 0.5944 0.6678]
PLSCanonical 의 정답률 :  [-2.2317 -2.3325 -2.8916 -2.1475 -1.4449]
PLSRegression 의 정답률 :  [0.8027 0.7662 0.5225 0.5972 0.735 ]
PassiveAggressiveRegressor 의 정답률 :  [ 0.2701 -2.1286  0.0731 -3.4999 -0.1681]
PoissonRegressor 의 정답률 :  [0.8568 0.819  0.6675 0.68   0.7541]
QuantileRegressor 의 정답률 :  [0.4134 0.4653 0.4055 0.2468 0.3155]
RANSACRegressor 의 정답률 :  [0.7384 0.29   0.4862 0.081  0.7268]
RadiusNeighborsRegressor 의 정답률 :  [nan nan nan nan nan]
RandomForestRegressor 의 정답률 :  [0.921  0.8501 0.8224 0.8863 0.9055]
RegressorChain 에서 오류떴어~
Ridge 의 정답률 :  [0.8098 0.8062 0.5811 0.6346 0.7226]
RidgeCV 의 정답률 :  [0.8113 0.8001 0.5889 0.6401 0.7236]
SGDRegressor 의 정답률 :  [-3.18251898e+25 -7.14071415e+25 -4.80875316e+26 -5.49992713e+26
 -3.58482760e+26]
SVR 의 정답률 :  [0.2348 0.3158 0.2412 0.0495 0.1402]
StackingRegressor 에서 오류떴어~
TheilSenRegressor 의 정답률 :  [0.7916 0.7259 0.6028 0.5593 0.7192]
TransformedTargetRegressor 의 정답률 :  [0.8111 0.7984 0.5903 0.6408 0.7233]
TweedieRegressor 의 정답률 :  [0.7377 0.7566 0.5652 0.5769 0.6309]
VotingRegressor 에서 오류떴어~



Diabets 데이터셋의 결과를 소개합니다~

ARDRegression 의 정답률 :  [0.4987 0.4877 0.5628 0.3773 0.5347]
AdaBoostRegressor 의 정답률 :  [0.3712 0.4597 0.4948 0.3781 0.448 ]
BaggingRegressor 의 정답률 :  [0.2704 0.4125 0.3788 0.3891 0.3669]
BayesianRidge 의 정답률 :  [0.5008 0.4843 0.5546 0.376  0.5307]
CCA 의 정답률 :  [0.487  0.4261 0.5524 0.2171 0.5076]
DecisionTreeRegressor 의 정답률 :  [-0.2337 -0.1454 -0.1114 -0.0072 -0.0269]
DummyRegressor 의 정답률 :  [-0.0002 -0.003  -0.     -0.0038 -0.0096]
ElasticNet 의 정답률 :  [ 0.0081  0.0064  0.0092  0.0041 -0.0008]
ElasticNetCV 의 정답률 :  [0.4307 0.4615 0.4913 0.3567 0.4567]
ExtraTreeRegressor 의 정답률 :  [ 0.1513 -0.0927 -0.1544  0.0497 -0.1202]
ExtraTreesRegressor 의 정답률 :  [0.3539 0.4973 0.5303 0.3917 0.4378]
GammaRegressor 의 정답률 :  [ 0.0052  0.0037  0.0061  0.0017 -0.0031]
GaussianProcessRegressor 의 정답률 :  [ -5.6361 -15.274   -9.9498 -12.4688 -12.0479]
GradientBoostingRegressor 의 정답률 :  [0.3884 0.4846 0.4859 0.3969 0.4436]
HistGradientBoostingRegressor 의 정답률 :  [0.289  0.4381 0.5171 0.3727 0.3564]
HuberRegressor 의 정답률 :  [0.5033 0.4751 0.5465 0.3688 0.5173]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.3968 0.3257 0.4331 0.3264 0.3547]
KernelRidge 의 정답률 :  [-3.3848 -3.4937 -4.0996 -3.3904 -3.6004]
Lars 의 정답률 :  [ 0.492  -0.6648 -1.0441 -0.0424  0.5119]
LarsCV 의 정답률 :  [0.4931 0.4877 0.5543 0.38   0.5241]
Lasso 의 정답률 :  [0.3432 0.3535 0.3859 0.3161 0.3605]
LassoCV 의 정답률 :  [0.498  0.4839 0.5593 0.3774 0.5164]
LassoLars 의 정답률 :  [0.3654 0.3781 0.4064 0.3364 0.3844]
LassoLarsCV 의 정답률 :  [0.4972 0.4843 0.5598 0.3798 0.5119]
LassoLarsIC 의 정답률 :  [0.4994 0.4911 0.5613 0.3794 0.5248]
LinearRegression 의 정답률 :  [0.5064 0.4868 0.5537 0.3794 0.5119]
LinearSVR 의 정답률 :  [-0.3347 -0.3163 -0.4191 -0.3029 -0.4732]
MLPRegressor 의 정답률 :  [-2.7427 -3.2046 -3.1733 -2.8523 -3.1844]
MultiOutputRegressor 에서 오류떴어~
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.1447 0.1735 0.1854 0.1389 0.1664]
OrthogonalMatchingPursuit 의 정답률 :  [0.3293 0.2857 0.3894 0.1967 0.3592]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.4785 0.4866 0.557  0.3704 0.5362]
PLSCanonical 의 정답률 :  [-0.9751 -1.6853 -0.8821 -1.3399 -1.1604]
PLSRegression 의 정답률 :  [0.4766 0.4763 0.5388 0.3819 0.5472]
PassiveAggressiveRegressor 의 정답률 :  [0.4556 0.4804 0.527  0.3414 0.4741]
PoissonRegressor 의 정답률 :  [0.3206 0.358  0.3666 0.282  0.3434]
QuantileRegressor 의 정답률 :  [-0.0219 -0.011  -0.0284 -0.0066 -0.0595]
RANSACRegressor 의 정답률 :  [0.0199 0.3447 0.131  0.1838 0.2176]
RadiusNeighborsRegressor 의 정답률 :  [-0.0002 -0.003  -0.     -0.0038 -0.0096]
RandomForestRegressor 의 정답률 :  [0.3745 0.4875 0.5011 0.3901 0.4263]
RegressorChain 에서 오류떴어~
Ridge 의 정답률 :  [0.4094 0.4479 0.4706 0.3447 0.4334]
RidgeCV 의 정답률 :  [0.4953 0.4876 0.5517 0.3802 0.5275]
SGDRegressor 의 정답률 :  [0.3933 0.442  0.4646 0.3296 0.415 ]
SVR 의 정답률 :  [0.1433 0.1844 0.1786 0.1425 0.1469]
StackingRegressor 에서 오류떴어~
TheilSenRegressor 의 정답률 :  [0.5064 0.4456 0.5485 0.3423 0.5167]
TransformedTargetRegressor 의 정답률 :  [0.5064 0.4868 0.5537 0.3794 0.5119]
TweedieRegressor 의 정답률 :  [ 0.0059  0.0043  0.007   0.0018 -0.0032]
VotingRegressor 에서 오류떴어~



Fetch_covtype 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  [0.5549 0.5619 0.5256 0.5009 0.5199]
BaggingClassifier 의 정답률 :  [0.9612 0.9624 0.9615 0.9623 0.962 ]
BernoulliNB 의 정답률 :  [0.6293 0.6318 0.6303 0.6337 0.6253]
'''