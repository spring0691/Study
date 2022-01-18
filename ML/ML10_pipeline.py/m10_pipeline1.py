from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
#1. 데이터 로드

datasets = load_iris()

x = datasets.data  
y = datasets.target
 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델링 모델구성

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC   
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import make_pipeline,Pipeline   # 이걸로 파이프라인을 만들어줄거다.

#model = SVC()
model = make_pipeline(MinMaxScaler(),SVC())           # 이런식으로 파이프라인을 만들어준다. 순서를 잘 지켜야한다.   GridSearch나 각종 Search안에 이걸 넣어준다.
                                                      # make_pipeline에서 test값까지 scaler해준다.
model.fit(x_train,y_train)     

#4. 평가, 예측

result = model.score(x_test,y_test)   

print("model.score : ", result)
