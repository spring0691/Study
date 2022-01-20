# 4개의 그래프가 한 화면에 나오게 만들어보기

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  
import warnings, numpy as np

warnings.filterwarnings(action='ignore')
#1. 데이터 로드

datasets = load_iris()

x = datasets.data  
y = datasets.target
#print(datasets.feature_names) 
#print(datasets.DESCR)  
#x = np.delete(x,0,axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier     # tree계열에서만 feature_importances가 있다.
from xgboost import XGBClassifier

model1 = DecisionTreeClassifier(max_depth=5,random_state=66)
model2 = RandomForestClassifier(max_depth=5,random_state=66)
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

#3. 컴파일,훈련
from sklearn.metrics import accuracy_score
                                    
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

#1.내가 젤 처음 한 방식. model1~4를 for문으로 불러오지 못했다. model.fit도 4번 따로해줘야한다.
'''
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plot_feature_importances_dataset(model1)
plt.subplot(2,2,2)
plot_feature_importances_dataset(model2)
plt.subplot(2,2,3)
plot_feature_importances_dataset(model3)
plt.subplot(2,2,4)
plot_feature_importances_dataset(model4)
plt.show()
'''                     


#2. model+i index번호 매기는건 변수 활성화가 안먹히고 model_list에 model들을 담아서 불러온다.

model_list = [model1,model2,model3,model4]
plt.figure(figsize=(20,20))
for i,model in enumerate(model_list,start=1):
    model.fit(x_train,y_train)
    plt.subplot(2,2,i)
    plot_feature_importances_dataset(model)
plt.show()

# result = model.score(x_test,y_test)
# y_pred = model.predict(x_test)
# acc = accuracy_score(y_test,y_pred)
# print("model.score : ",result)
# print("accuracy_scroe : ",acc)        이 부분은 그냥 확인용. 
