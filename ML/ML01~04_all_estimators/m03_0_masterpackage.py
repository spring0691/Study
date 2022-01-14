from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC   
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
import warnings
warnings.filterwarnings(action='ignore')

# 회귀 & 분류 섞여있다 try except로 빼주고 for문 돌린다

dd =  [load_iris(),load_breast_cancer(),load_wine(),load_boston(),load_diabetes(),fetch_covtype()]
#load_iris      <class 'function'>
#load_iris()    <class 'sklearn.utils.Bunch'>

mm = [Perceptron(),LinearSVC(),SVC(),KNeighborsClassifier(),KNeighborsRegressor(),LinearRegression(),LogisticRegression(),DecisionTreeClassifier(),DecisionTreeRegressor(),RandomForestClassifier(),RandomForestRegressor()]

for i,data in enumerate(dd,start=1):
    datasets = data

    x = datasets.data  
    y = datasets.target

    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
    print(f'{i}번째 데이터셋의 결과를 소개합니다~\n')
    
    for md in mm:
        model = md
        try:
            model.fit(x_train,y_train)    
            result = model.score(x_test,y_test) 
            print(f"{md}_score : ", result)
        except:
            print(f'{i}번째 데이터셋의 {model}에서 에러가 터졌습니다~')
    print('\n\n')
            
         
     
    
