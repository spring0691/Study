from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score,accuracy_score
import numpy as np,pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes()} # ,fetch_covtype()
scaler = MinMaxScaler()
classifier_all = all_estimators(type_filter='classifier')  
regressor_all = all_estimators(type_filter='regressor')

'''     kaggle bike 데이터
path = 'D:\Project\Kaggle_Project\bike'
train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')   
'''

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
                print(cn,'에서 오류떴어~')
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
                print(rn,'에서 오류떴어~')
        print('\n\n')

'''
Iirs 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  0.6333333333333333
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.4
CalibratedClassifierCV 의 정답률 :  0.9666666666666667
CategoricalNB 의 정답률 :  0.3333333333333333
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  0.6666666666666666
DecisionTreeClassifier 의 정답률 :  0.9333333333333333
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  0.9666666666666667
ExtraTreesClassifier 의 정답률 :  0.9666666666666667
GaussianNB 의 정답률 :  0.9666666666666667
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9666666666666667
HistGradientBoostingClassifier 의 정답률 :  0.8666666666666667
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  0.9666666666666667
LabelSpreading 의 정답률 :  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9666666666666667
LogisticRegression 의 정답률 :  0.9666666666666667
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.9666666666666667
MultiOutputClassifier 에서 오류떴어~
MultinomialNB 의 정답률 :  0.6333333333333333
NearestCentroid 의 정답률 :  0.9666666666666667
NuSVC 의 정답률 :  0.9666666666666667
OneVsOneClassifier 에서 오류떴어~
OneVsRestClassifier 에서 오류떴어~
OutputCodeClassifier 에서 오류떴어~
PassiveAggressiveClassifier 의 정답률 :  0.7666666666666667
Perceptron 의 정답률 :  0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.4666666666666667
RandomForestClassifier 의 정답률 :  0.9333333333333333
RidgeClassifier 의 정답률 :  0.9333333333333333
RidgeClassifierCV 의 정답률 :  0.8333333333333334
SGDClassifier 의 정답률 :  0.9
SVC 의 정답률 :  1.0
StackingClassifier 에서 오류떴어~
VotingClassifier 에서 오류떴어~



Breast_cancer 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  0.9473684210526315
BaggingClassifier 의 정답률 :  0.9649122807017544
BernoulliNB 의 정답률 :  0.6403508771929824
CalibratedClassifierCV 의 정답률 :  0.9649122807017544
CategoricalNB 에서 오류떴어~
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  0.7807017543859649
DecisionTreeClassifier 의 정답률 :  0.9385964912280702
DummyClassifier 의 정답률 :  0.6403508771929824
ExtraTreeClassifier 의 정답률 :  0.8947368421052632
ExtraTreesClassifier 의 정답률 :  0.9649122807017544
GaussianNB 의 정답률 :  0.9210526315789473
GaussianProcessClassifier 의 정답률 :  0.9649122807017544
GradientBoostingClassifier 의 정답률 :  0.956140350877193
HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
KNeighborsClassifier 의 정답률 :  0.956140350877193
LabelPropagation 의 정답률 :  0.9473684210526315
LabelSpreading 의 정답률 :  0.9473684210526315
LinearDiscriminantAnalysis 의 정답률 :  0.9473684210526315
LinearSVC 의 정답률 :  0.9736842105263158
LogisticRegression 의 정답률 :  0.9649122807017544
LogisticRegressionCV 의 정답률 :  0.9736842105263158
MLPClassifier 의 정답률 :  0.9736842105263158
MultiOutputClassifier 에서 오류떴어~
MultinomialNB 의 정답률 :  0.8508771929824561
NearestCentroid 의 정답률 :  0.9298245614035088
NuSVC 의 정답률 :  0.9473684210526315
OneVsOneClassifier 에서 오류떴어~
OneVsRestClassifier 에서 오류떴어~
OutputCodeClassifier 에서 오류떴어~
PassiveAggressiveClassifier 의 정답률 :  0.9473684210526315
Perceptron 의 정답률 :  0.9736842105263158
QuadraticDiscriminantAnalysis 의 정답률 :  0.9385964912280702
RadiusNeighborsClassifier 에서 오류떴어~
RandomForestClassifier 의 정답률 :  0.9736842105263158
RidgeClassifier 의 정답률 :  0.9473684210526315
RidgeClassifierCV 의 정답률 :  0.9473684210526315
SGDClassifier 의 정답률 :  0.9912280701754386
SVC 의 정답률 :  0.9736842105263158
StackingClassifier 에서 오류떴어~
VotingClassifier 에서 오류떴어~



Wine 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  0.8888888888888888
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.4166666666666667
CalibratedClassifierCV 의 정답률 :  0.9722222222222222
CategoricalNB 의 정답률 :  0.5
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  0.8611111111111112
DummyClassifier 의 정답률 :  0.4166666666666667
ExtraTreeClassifier 의 정답률 :  0.75
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  1.0
GradientBoostingClassifier 의 정답률 :  0.9722222222222222
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  1.0
LabelSpreading 의 정답률 :  1.0
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9722222222222222
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  0.9722222222222222
MLPClassifier 의 정답률 :  1.0
MultiOutputClassifier 에서 오류떴어~
MultinomialNB 의 정답률 :  0.9444444444444444
NearestCentroid 의 정답률 :  1.0
NuSVC 의 정답률 :  1.0
OneVsOneClassifier 에서 오류떴어~
OneVsRestClassifier 에서 오류떴어~
OutputCodeClassifier 에서 오류떴어~
PassiveAggressiveClassifier 의 정답률 :  0.9722222222222222  
Perceptron 의 정답률 :  0.9722222222222222
QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
RadiusNeighborsClassifier 의 정답률 :  0.9722222222222222    
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  1.0
RidgeClassifierCV 의 정답률 :  0.9722222222222222
SGDClassifier 의 정답률 :  0.9722222222222222    
SVC 의 정답률 :  1.0
StackingClassifier 에서 오류떴어~
VotingClassifier 에서 오류떴어~



Boston 데이터셋의 결과를 소개합니다~

ARDRegression 의 정답률 :  0.8119016106669674
AdaBoostRegressor 의 정답률 :  0.8756574637507429
BaggingRegressor 의 정답률 :  0.9056720532391974
BayesianRidge 의 정답률 :  0.8119880571377842
CCA 의 정답률 :  0.791347718442463
DecisionTreeRegressor 의 정답률 :  0.7697414461744618
DummyRegressor 의 정답률 :  -0.0005370164400797517
ElasticNet 의 정답률 :  0.16201563080833714
ElasticNetCV 의 정답률 :  0.8113737663385278
ExtraTreeRegressor 의 정답률 :  0.7990290167554419
ExtraTreesRegressor 의 정답률 :  0.9389621584129879
GammaRegressor 의 정답률 :  0.1964792057029865
GaussianProcessRegressor 의 정답률 :  -1.578958693000398
GradientBoostingRegressor 의 정답률 :  0.9452784393322317
HistGradientBoostingRegressor 의 정답률 :  0.9323326124661162
HuberRegressor 의 정답률 :  0.7958373063951082
IsotonicRegression 에서 오류떴어~
KNeighborsRegressor 의 정답률 :  0.8265307833211177
KernelRidge 의 정답률 :  0.803254958502079
Lars 의 정답률 :  0.7746736096721598
LarsCV 의 정답률 :  0.7981576314184016
Lasso 의 정답률 :  0.242592140544296
LassoCV 의 정답률 :  0.8125908596954046
LassoLars 의 정답률 :  -0.0005370164400797517
LassoLarsCV 의 정답률 :  0.8127604328474283
LassoLarsIC 의 정답률 :  0.8131423868817642
LinearRegression 의 정답률 :  0.8111288663608667
LinearSVR 의 정답률 :  0.7086540596052056
MLPRegressor 의 정답률 :  0.44220020724152154
MultiOutputRegressor 에서 오류떴어~
MultiTaskElasticNet 에서 오류떴어~
MultiTaskElasticNetCV 에서 오류떴어~
MultiTaskLasso 에서 오류떴어~
MultiTaskLassoCV 에서 오류떴어~
NuSVR 의 정답률 :  0.6254681434531
OrthogonalMatchingPursuit 의 정답률 :  0.5827617571381449
OrthogonalMatchingPursuitCV 의 정답률 :  0.78617447738729
PLSCanonical 의 정답률 :  -2.2317079741425734
PLSRegression 의 정답률 :  0.8027313142007888
PassiveAggressiveRegressor 의 정답률 :  0.6976428108715289
PoissonRegressor 의 정답률 :  0.67496007101481
QuantileRegressor 의 정답률 :  -0.020280478327147522
RANSACRegressor 의 정답률 :  0.7404826886723466
RadiusNeighborsRegressor 의 정답률 :  0.41191760158788593
RandomForestRegressor 의 정답률 :  0.9242899656217718
RegressorChain 에서 오류떴어~
Ridge 의 정답률 :  0.8087497007195745
RidgeCV 의 정답률 :  0.8116598578372426
SGDRegressor 의 정답률 :  0.8230618453847427
SVR 의 정답률 :  0.6597910766772523
StackingRegressor 에서 오류떴어~
TheilSenRegressor 의 정답률 :  0.7912524096958482
TransformedTargetRegressor 의 정답률 :  0.8111288663608667
TweedieRegressor 의 정답률 :  0.19473445117356525
VotingRegressor 에서 오류떴어~



Diabets 데이터셋의 결과를 소개합니다~

ARDRegression 의 정답률 :  0.498748289056254
AdaBoostRegressor 의 정답률 :  0.3708217274314811
BaggingRegressor 의 정답률 :  0.36960313463317973
BayesianRidge 의 정답률 :  0.5014366863847451
CCA 의 정답률 :  0.48696409064967594
DecisionTreeRegressor 의 정답률 :  -0.15987055223925029
DummyRegressor 의 정답률 :  -0.00015425885559339214
ElasticNet 의 정답률 :  0.11987522766332959
ElasticNetCV 의 정답률 :  0.48941369735908524
ExtraTreeRegressor 의 정답률 :  -0.23106693736588824
ExtraTreesRegressor 의 정답률 :  0.3821760293096682
GammaRegressor 의 정답률 :  0.07219655012236648
GaussianProcessRegressor 의 정답률 :  -7.547010959777328
GradientBoostingRegressor 의 정답률 :  0.39173242770343875
HistGradientBoostingRegressor 의 정답률 :  0.28899497703380905
HuberRegressor 의 정답률 :  0.5068530513878713
IsotonicRegression 에서 오류떴어~
KNeighborsRegressor 의 정답률 :  0.3741821819765594
KernelRidge 의 정답률 :  0.48022687224693617
Lars 의 정답률 :  0.4919866521464151
LarsCV 의 정답률 :  0.5010892359535754
Lasso 의 정답률 :  0.46430753276688697
LassoCV 의 정답률 :  0.4992382182931273
LassoLars 의 정답률 :  0.3654388741895792
LassoLarsCV 의 정답률 :  0.4951942790678243
LassoLarsIC 의 정답률 :  0.49940515175310685
LinearRegression 의 정답률 :  0.5063891053505036
LinearSVR 의 정답률 :  0.14945390399691316
MLPRegressor 의 정답률 :  -0.5272045075956702
MultiOutputRegressor 에서 오류떴어~
MultiTaskElasticNet 에서 오류떴어~
MultiTaskElasticNetCV 에서 오류떴어~
MultiTaskLasso 에서 오류떴어~
MultiTaskLassoCV 에서 오류떴어~
NuSVR 의 정답률 :  0.12527149380257419
OrthogonalMatchingPursuit 의 정답률 :  0.3293449115305741
OrthogonalMatchingPursuitCV 의 정답률 :  0.44354253337919725
PLSCanonical 의 정답률 :  -0.9750792277922931
PLSRegression 의 정답률 :  0.4766139460349792
PassiveAggressiveRegressor 의 정답률 :  0.4887644407046796
PoissonRegressor 의 정답률 :  0.4823231874912104
QuantileRegressor 의 정답률 :  -0.021939242070499576
RANSACRegressor 의 정답률 :  0.10460616421903557
RadiusNeighborsRegressor 의 정답률 :  0.14407236562185122
RandomForestRegressor 의 정답률 :  0.38770638909492194
RegressorChain 에서 오류떴어~
Ridge 의 정답률 :  0.49950383964954104
RidgeCV 의 정답률 :  0.49950383964954104
SGDRegressor 의 정답률 :  0.49500263771640174
SVR 의 정답률 :  0.12343791188320263
StackingRegressor 에서 오류떴어~
TheilSenRegressor 의 정답률 :  0.5146530047901914
TransformedTargetRegressor 의 정답률 :  0.5063891053505036
TweedieRegressor 의 정답률 :  0.07335459385974419
VotingRegressor 에서 오류떴어~



Fetch_covtype 데이터셋의 결과를 소개합니다~

AdaBoostClassifier 의 정답률 :  0.5028613719095032
BaggingClassifier 의 정답률 :  0.9624020033906181
BernoulliNB 의 정답률 :  0.631833945767321
CalibratedClassifierCV 의 정답률 :  0.7122621619063191
CategoricalNB 의 정답률 :  0.6321437484402296
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  0.6225742880992745
DecisionTreeClassifier 의 정답률 :  0.9393819436675473
DummyClassifier 의 정답률 :  0.48625250638968015
ExtraTreeClassifier 의 정답률 :  0.8606662478593495
ExtraTreesClassifier 의 정답률 :  0.9542180494479489
GaussianNB 의 정답률 :  0.09079800005163378
GaussianProcessClassifier 에서 오류떴어~
GradientBoostingClassifier 의 정답률 :  0.773491217954786
HistGradientBoostingClassifier 의 정답률 :  0.7799540459368519
KNeighborsClassifier 의 정답률 :  0.9376263951877318
LabelPropagation 에서 오류떴어~
LabelSpreading 에서 오류떴어~
LinearDiscriminantAnalysis 의 정답률 :  0.6797931206595355
LinearSVC 의 정답률 :  0.7124084576129704
LogisticRegression 의 정답률 :  0.7194220459024294
'''