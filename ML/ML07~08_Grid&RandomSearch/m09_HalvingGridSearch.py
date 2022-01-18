from sklearn.datasets import load_iris
from sklearn.svm import SVC   
import numpy as np, pandas as pd
import warnings
from sklearn.metrics import accuracy_score,r2_score
warnings.filterwarnings(action='ignore')

datasets = load_iris()

x = datasets.data
y = datasets.target

# Halving모델은 아직 sklearn에서 완성되지 않아서 추가로 다른것을 import해줘야한다.
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=66,train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = { "C":[1, 10, 100, 1000], "kernel":["linear","rbf","sigmoid"], "gamma":[0.01, 0.001, 0.0001,0.00001], "degree":[3,4,5,6]}

#2. 모델구성

#model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,n_jobs=4, refit=True ) #, n_jobs=-1
#model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1,n_jobs=-1, refit=True, n_iter = 20, random_state=66) #, n_jobs=-1
model = HalvingGridSearchCV(SVC(), parameters, cv=kfold, verbose=1,n_jobs=-1, refit=True ) 


#3. 훈련

import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

#4. 평가,예측

# x_test = x_train  # 과적합 상황 보여주기
# y_test = y_train  # train데이터로 best_estimator_로 예측뒤 점수를 내면 
                    # best_score_ 나온다.

print("최적의 매개변수 : ", model.best_estimator_)  # 최적의 평가자 평가측정?   SVC(C=1, gamma=0.01, kernel='linear') 
print("최적의 파라미터 : ", model.best_params_)     # 최적의 파라미터도 보여준다 {'C': 1, 'degree': 3, 'gamma': 0.01, 'kernel': 'linear'}  한번 측정하고 다음부터는 이거로 쓰면된당.

print("best_score_ : ", model.best_score_)             # 가장 좋은 값을 보여준다.     train & validation만 가지고 나온 값.               best_score_ :  0.9916666666666668
print("model.score : ", model.score(x_test,y_test))    # .score가 evaulate개념이다.  train & validation -> 하고 나서 test 한 후에 나온값 model.score :  0.9666666666666667

y_pred = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)         
print("최적 튠 ACC : ", accuracy_score(y_test,y_pred_best))     # 각기 다른방식으로 값을 뽑아봄으로써 model에서 최고값을 주는지 평균값을 주는지 확인할 수 있다.
print("걸린시간 : ", end - start)


