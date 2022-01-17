# grid Search Cross Validation의 약자이다.  모든 경우의 수를 전부 실행함으로써 최적의 1값만을 뽑아내서 저장한다. One for All. 하나의 최적을 위해서 모든 경우의 수를 다 계산한다.

from tabnanny import verbose
from sklearn.datasets import load_iris
from sklearn.svm import SVC   
import numpy as np
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
model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1) # 여기 단계에서 cross validation까지 때려버린다  GridSearchCV로 래핑해준거다.  verbose로 내용볼수있다.

#3. 훈련

model.fit(x_train,y_train)

#4. 평가,예측



print("최적의 매개변수 : ", model.best_estimator_)  # 최적의 평가자 평가측정?   SVC(C=1, gamma=0.01, kernel='linear') 
print("최적의 파라미터 : ", model.best_params_)     # 최적의 파라미터도 보여준다 {'C': 1, 'degree': 3, 'gamma': 0.01, 'kernel': 'linear'}  한번 측정하고 다음부터는 이거로 쓰면된당.

print("best_score_ : ", model.best_score_)             # 가장 좋은 값을 보여준다.     train & validation만 가지고 나온 값.               best_score_ :  0.9916666666666668
print("model.score : ", model.score(x_test,y_test))    # .score가 evaulate개념이다.  train & validation -> 하고 나서 test 한 후에 나온값 model.score :  0.9666666666666667

y_pred = model.predict(x_test)

print("acc_score : ", accuracy_score(y_test,y_pred))
#scores = cross_val_score(model,x_train,y_train,cv=kfold)
#print("ACC : ", np.round(scores,4),"\ncross_val_score : ",round(np.mean(scores),4))

#print(model.cv)    # KFold(n_splits=5, random_state=66, shuffle=True)