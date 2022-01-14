from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings(action='ignore')
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

allAlgorithms = all_estimators(type_filter='classifier')  # 모든 분류모델들 하나로 묶어줌 짱짱 와우~    41개
#allAlgorithms = all_estimators(type_filter='regressor')    # 모든 회귀모델들 하나로 묶어줌 짱짱 와우~  55개
#print('allAlgorithms : ',allAlgorithms)        모델과 위치가 들어있는 딕셔너리들이 리스트에 들어있다
#print('모델의 갯수 : ', len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        print(name,'에서 오류떴어~')
   
'''
AdaBoostClassifier 의 정답률 :  0.6333333333333333
BaggingClassifier 의 정답률 :  0.9333333333333333
BernoulliNB 의 정답률 :  0.4
CalibratedClassifierCV 의 정답률 :  0.9666666666666667
CategoricalNB 의 정답률 :  0.3333333333333333
ClassifierChain 에서 오류떴어~
ComplementNB 의 정답률 :  0.6666666666666666
DecisionTreeClassifier 의 정답률 :  0.9666666666666667
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  0.9
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
PassiveAggressiveClassifier 의 정답률 :  0.7
Perceptron 의 정답률 :  0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.4666666666666667
RandomForestClassifier 의 정답률 :  0.9333333333333333
RidgeClassifier 의 정답률 :  0.9333333333333333
RidgeClassifierCV 의 정답률 :  0.8333333333333334
SGDClassifier 의 정답률 :  0.8
SVC 의 정답률 :  1.0
StackingClassifier 에서 오류떴어~
VotingClassifier 에서 오류떴어~
'''