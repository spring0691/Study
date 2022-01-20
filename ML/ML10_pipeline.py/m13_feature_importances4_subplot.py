# 4개의 그래프가 한 화면에 나오게 만들어보기

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

model1 = DecisionTreeClassifier(max_depth=5,random_state=66)
model2 = RandomForestClassifier(max_depth=5,random_state=66)
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()

#3. 컴파일, 훈련

# model1.fit(x_train,y_train)     
# model2.fit(x_train,y_train)     
# model3.fit(x_train,y_train)     
# model4.fit(x_train,y_train)     

#4. 평가, 예측
from sklearn.metrics import accuracy_score
# result1 = model1.score(x_test,y_test)   
# result2 = model2.score(x_test,y_test)   
# result3 = model3.score(x_test,y_test)   
# result4 = model4.score(x_test,y_test)   

# print("model_score : ", result1)
# print("model_score : ", result2)
# print("model_score : ", result3)
# print("model_score : ", result4)

# print(model1.feature_importances_)   #[0.         0.0125026  0.53835801 0.44913938]  4개의 피쳐중에 뭐가 중요한지 보여준다.
# print(model2.feature_importances_)   #[0.         0.0125026  0.53835801 0.44913938]  4개의 피쳐중에 뭐가 중요한지 보여준다.
# print(model3.feature_importances_)   #[0.         0.0125026  0.53835801 0.44913938]  4개의 피쳐중에 뭐가 중요한지 보여준다.
# print(model4.feature_importances_)   #[0.         0.0125026  0.53835801 0.44913938]  4개의 피쳐중에 뭐가 중요한지 보여준다.
                                    # fit 돌리고 나서 각 피쳐의 연관성을 보여준다.
                                    
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

model_list = [model1,model2,model3,model4]
plt.figure(figsize=(15,15))
for i,model in enumerate(model_list,start=1):
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    print("model.score : ",result)
    print("accuracy_scroe : ",acc)
    plt.subplot(2,2,i)
    plot_feature_importances_dataset(model)
plt.show()

