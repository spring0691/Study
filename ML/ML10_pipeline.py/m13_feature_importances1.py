# feature_importances -> 본격적으로 작업을 들어가기전에 가볍게 default설정으로 one cycle돌려보고 해당 모델을 돌릴때
# 그다지 중요하지 않다고 생각되는 feature를 한번 훑어보고 미리 솎아내주는 개념

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  
import warnings

warnings.filterwarnings(action='ignore')
#1. 데이터 로드

datasets = load_iris()

x = datasets.data  
y = datasets.target
#print(datasets.feature_names) 
#print(datasets.DESCR)  
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier     # tree계열에서만 feature_importances가 있다.
from xgboost import XGBClassifier

#model = DecisionTreeClassifier(max_depth=5,random_state=66)
#model = RandomForestClassifier(max_depth=5,random_state=66)
#model = XGBClassifier()
model = GradientBoostingClassifier()

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
model_score :  0.9666666666666667
[0.         0.0125026  0.53835801 0.44913938]

RandomForestClassifier
model_score :  0.9666666666666667
[0.08150824 0.02190985 0.46987909 0.42670282]

XGBClassifier
model_score :  0.9
[0.01835513 0.0256969  0.6204526  0.33549538]

GradientBoostingClassifier
model_score :  0.9333333333333333
[0.00284751 0.01240858 0.23336692 0.75137699]
'''