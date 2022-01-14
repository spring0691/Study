from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression     
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

# Train test 나누면 test데이터 자체를 못쓰니까 kfold로 골고루 섞어줘서 모든데이터 활용할수있게한다~

model = Perceptron(),LinearSVC(),SVC(),KNeighborsClassifier(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()

dd =  {'Iirs':load_iris(),'Breast_cancer':load_breast_cancer(),'Wine':load_wine(),'Boston':load_boston(),'Diabets':load_diabetes()} # ,'Fetch_covtype':fetch_covtype()

datasets = load_iris()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split,KFold, cross_val_score

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=66,train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = SVC()

scores = cross_val_score(model,x_train,y_train,cv=kfold)

print("ACC : ", np.round(scores,4),"\n cross_val_score : ",round(np.mean(scores),4))




