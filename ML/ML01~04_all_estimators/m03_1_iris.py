from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  
import warnings

warnings.filterwarnings(action='ignore')
#1. 데이터 로드

datasets = load_iris()

x = datasets.data  
y = datasets.target
 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 모델구성

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     # 이름은 Regression인데 분류에서 쓰인다? 분류모델 착각하기 쉽다.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


model = Perceptron(),LinearSVC(),SVC(),KNeighborsClassifier(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()

for i in model:    
    model = i
    
    #3. 컴파일, 훈련

    model.fit(x_train,y_train)     

    #4. 평가, 예측

    result = model.score(x_test,y_test)   

    y_predict = model.predict(x_test)

    print(f"{i} : ", result)
    