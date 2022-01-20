from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.datasets import load_iris  
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np,time
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
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import make_pipeline,Pipeline   
from sklearn.decomposition import PCA                 
     
# parameters = [{'randomforestclassifier__n_estimators' : [100,200], 'randomforestclassifier__max_depth' : [6, 8, 10, 12], 
#                'randomforestclassifier__min_samples_leaf' : [3, 5, 7, 10], 'randomforestclassifier__min_samples_split' : [3, 5, 7, 10]}]    

parameters = [{'Rfcl__n_estimators' : [100,200], 'Rfcl__max_depth' : [6, 8, 10, 12], 
               'Rfcl__min_samples_leaf' : [3, 5, 7, 10], 'Rfcl__min_samples_split' : [3, 5, 7, 10]}]      

# parameters를 Pipeline이 알아먹도록 형태를 바꿔줘야한다. 
#각 파라미터 이름앞에 모델 이름의 명시해준다. randomforestclassifier__ 이런식으로 엮어준다. <-- pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 이 방식에서 씀
#모델이름 명시할때 조금이라도 줄여쓸수 있다. Rfcl__ 이 방식은 <-- pipe = Pipeline([( "mm",MinMaxScaler()),("Rfcl",RandomForestClassifier(n_jobs=-1))]) 사전에 정의해줘야한다.

#pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([( "mm",MinMaxScaler()),("Rfcl",RandomForestClassifier(random_state=66))])      

model = GridSearchCV(pipe,parameters,cv=5,verbose=1, n_jobs=-1)  # 모델자리엔 Pipeline을 넣었는데 parameters자리엔 RandomForest에 대한 parameters가 있어서 아구가 안맞다.n_jobs는 제일 바깥쪽에 써주는게 좋다.
                                                                       
start = time.time()                                                                      
model.fit(x_train,y_train)     
end = time.time()

#4. 평가, 예측

result = model.score(x_test,y_test)   

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)

print('걸린시간은요 ~ : ', end - start)
print("model.score : ", result)
print("acc_score는요 ~ : ", acc)
