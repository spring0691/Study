# 13_1을 가져와서 feature 하나 제거하고 1번과 성능비교.
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
#1. 데이터 로드

datasets = load_iris()

x = datasets.data  
y = datasets.target
#print(datasets.feature_names)   ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 첫번째를 제거할것이다!
#print(datasets.DESCR)  

x = np.delete(x,[0,1],axis=1)   # pandas의 drop과 똑같은 기능 제거할 데이터, index번호(0부터 시작), 그리고 행과 열을 지정해주면 끝! 깔끔

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier     # tree계열에서만 feature_importances가 있다.
from xgboost import XGBClassifier    

#model = DecisionTreeClassifier(max_depth=5,random_state=66)
#model = RandomForestClassifier(max_depth=5,random_state=66)
model = XGBClassifier(random_state=66,eval_metric='error')
#model = GradientBoostingClassifier()

#3. 컴파일, 훈련


model.fit(x_train,y_train)     

#4. 평가, 예측
from sklearn.metrics import accuracy_score
result = model.score(x_test,y_test)   

print("model_score : ", result)

print(model.feature_importances_)   #[0.         0.0125026  0.53835801 0.44913938]  4개의 피쳐중에 뭐가 중요한지 보여준다.
                                    # fit 돌리고 나서 각 피쳐의 정확도를 보여준다.

'''
DecisionTreeClassifier
model_score :  0.9333333333333333
[0.54517411 0.45482589]

RandomForestClassifier
model_score :  0.9666666666666667
[0.51997516 0.48002484]

XGBClassifier
model_score :  0.9666666666666667
[0.51089597 0.489104  ]

GradientBoostingClassifier
model_score :  0.9666666666666667
[0.30373614 0.69626386]
'''