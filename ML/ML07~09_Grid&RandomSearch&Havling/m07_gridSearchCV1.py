# grid Search Cross Validation의 약자이다.  모든 경우의 수를 전부 실행함으로써 최적의 1값만을 뽑아내서 저장한다. One for All. 하나의 최적을 위해서 모든 경우의 수를 다 계산한다.

from sklearn.datasets import load_iris
from sklearn.svm import SVC   
import numpy as np, pandas as pd
import warnings
from sklearn.metrics import accuracy_score,r2_score
warnings.filterwarnings(action='ignore')

datasets = load_iris()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=66,train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

'''
parameters = [
            {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},        # SVC(C=1, kernel='linear', degree=3) 이런거처럼 모든 경우의 수를 parameters에 담아준다. 12
            {"C":[1, 10, 100], "kernel":["rbf"],"gamma":[0.001, 0.0001]},           # 6
            {"C":[1, 10, 100, 1000],"kernel":["sigmoid"],"gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}    # 24
]                                                                                                           # 총 42번
'''

parameters = [{ "C":[1, 10, 100, 1000], "kernel":["linear","rbf","sigmoid"], "gamma":[0.01, 0.001, 0.0001], "degree":[3,4,5]}]
#2. 모델구성

model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,n_jobs=4, refit=True ) #, n_jobs=-1
# 여기 단계에서 cross validation까지 때려버린다  GridSearchCV로 래핑해준거다.  verbose로 내용볼수있다. refit=True 여기서 best값 줄지 말지 결정한다.
# 병렬 CPU지원을 한다.  이 기능을 활용하면 병렬식으로 여러개 사용해서 더 빠르게 작업 할 수 있다. n_jobs = 1~ 내 장비가 사용가능한 값까지. -1하면 내 장비의 코어 다 사용한다

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
#print("걸린시간 : ", float(end) - float(start))
############################################################################################
'''
Fitting 5 folds for each of 108 candidates, totalling 540 fits
최적의 매개변수 :  SVC(C=1, gamma=0.01, kernel='linear')
최적의 파라미터 :  {'C': 1, 'degree': 3, 'gamma': 0.01, 'kernel': 'linear'}
best_score_ :  0.9916666666666668
model.score :  0.9666666666666667
acc_score :  0.9666666666666667
최적 튠 ACC :  0.9666666666666667

#print(model.cv_results_)   cv의 결과값을 dict형태로 볼 수 있다.
aaa = pd.DataFrame(model.cv_results_)       # 1번의 cv에대한 결과

bbb = aaa[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score','split4_test_score']]
#           파라미터        평균값         파라미터의 등수      cv=kfold각 폴드의 값들.     캬 깔끔하네
ccc = aaa[['params','rank_test_score']]             
print(ccc)

# 내가 쓰고싶은 모델의 파라미터 종류를 확인해보고 어떻게 작동하는지 확인해보고 params에 지정한 후에 GridSearch돌려서 best값 찾을수 있다. 
# 말이 GridSearch지 그냥 모든 파라미터값 싹 다 자원써서 돌려보고 제일 좋은 1개값 기록해주는거다..ㅋㅋ 사람의 센스로 값 바꾸는거만큼 빠르지는 못하다.
'''