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

from sklearn.decomposition import PCA                 # 분해하다? Principal component analysis; PCA 주성분 분석 고차원의 데이터를 저차원의 데이터로 환원시키는 기법을 말한다.
                                                      # 선을하나 쫙 긋고 비슷한 놈들을 거리에 맞춰서 재조정한다. 내가 원하는 개수만큼 데이터를 줄여준다? 
                                                      # 성능을 올려주지는 않지만 차원을 축소시킴으로써 살짝의 정확성 저하에 비해 시간을 많이 줄일수 있다.

#model = make_pipeline(MinMaxScaler(), PCA(), RandomForestClassifier())   # 이런식으로 파이프라인을 만들어준다. 순서를 잘 지켜야한다.   GridSearch나 각종 Search안에 이걸 넣어준다.
# make_pipeline에서 test값까지 scaler해준다. 여러개 넣을수도있다.  중간에 차원축소를 시켜줌으로써 연산량을 감소시킨다.           
model = Pipeline([("mm",MinMaxScaler()),("PCA",PCA()),("Rfcl",RandomForestClassifier())])      # 사용법은 같은데 문법이 살짝 틀리다.
#이런식으로 큰 [ ]리스트 안에 작은 묶음 ( )로 묶고 지칭할 이름과 사용 api를 묶어서 넣어준다.
                                                                     
model.fit(x_train,y_train)     

#4. 평가, 예측

result = model.score(x_test,y_test)   

print("model.score : ", result)
