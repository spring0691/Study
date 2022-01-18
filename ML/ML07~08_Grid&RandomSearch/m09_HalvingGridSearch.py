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

'''
n_iterations: 2
n_required_iterations: 5
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 192
n_resources: 30
Fitting 5 folds for each of 192 candidates, totalling 960 fits
----------
iter: 1
n_candidates: 64
n_resources: 90
Fitting 5 folds for each of 64 candidates, totalling 320 fits
최적의 매개변수 :  SVC(C=1000, degree=6, gamma=0.001, kernel='sigmoid')
최적의 파라미터 :  {'C': 1000, 'degree': 6, 'gamma': 0.001, 'kernel': 'sigmoid'}
best_score_ :  0.9777777777777779
model.score :  0.9666666666666667
acc_score :  0.9666666666666667
최적 튠 ACC :  0.9666666666666667
걸린시간 :  1.7391059398651123

Halving 모델은 기존의 GridSearch에서 모든걸 다 카운트하면서 세는 방식을 좀 보완한것인데 randomsearch가 dropout처럼 줄였다면 Halving은 2등분한다? 오히려 X2 한다?
그러면 GridSearch보다 2배더 시간 걸리고 해야하는데 1번 돌때 상위 일정 %의 상위 값을 뽑고 2번째 돌때는 미리 뽑아놓은 값들로만 돌린다. 
1번 돌릴때 전체를 일반 Grid처럼 싹 다 돌리는게 아니라 여기저기 조합해서 좀 뽑아써서 1번 돌릴때 일반 Grid의 100%까지는 다 안하고 그 이하로 돌린다. 그 후 위의 내용을 실행한다.
HalvingGrid -> 데이터(params * cv)의 일부만 쓰겠다 일반Grid말고 HalvingGrid / Random -> 파라미터의 일부만 쓰겠다.   HalvingGrid와 Random은 살짝 다르다 그래서 

중첩교차검증? 
교차검증을 또 중첩하겠다 train & test로 일단 나눠서 그냥 실행 시킨후? train을 cv한다?

'''
