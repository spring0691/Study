# PCA -> n_component > 0.95 이상 xgboost, gridSearch 또는 RandomSearch를 쓸것 m17_2결과 뛰어넘기
from tabnanny import verbose
from tensorflow.keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import numpy as np,pandas as pd,time,sys, warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#0. 출력 관련 옵션들
warnings.filterwarnings(action='ignore')
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 190)

#1. 데이터 로드 및 정제

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train),-1)  # (60000,784)
x_test = x_test.reshape(len(x_test),-1)


pca_name_dict = {'0.95':154,'0.99':331,'0.999':486}


#2. 모델 설정

parameters = {"XGB__n_estimators":[100,200,300], "XGB__learning_rate":[0.1,0.001,0.01],"XGB__max_depth":[4,5,6],'XGB__use_label_encoder':[False],
            "XGB__colsample_bytree":[0.6,0.9,1],"XGB__colsample_bylevel":[0.6,0.7,0.9],"XGB__random_state":[66],"XGB__eval_metric":['error']} 

pipe_model = Pipeline([("mm",MinMaxScaler()),("PCA",PCA(n_components=154)),("XGB",XGBClassifier())])  # random_state=66,eval_metric='error'

model = RandomizedSearchCV(pipe_model,parameters,cv=3,n_jobs=-1,verbose=3)

start = time.time()
model.fit(x_train,y_train)
end = time.time()

print("최적의 매개변수는요~ : ", model.best_estimator_)
print('최적의 파라미터는요~ : ', model.best_params_)
print('model.score로 구한 값은요~ : ', model.score(x_test,y_test))   
print('걸린 시간은요~ : ', end - start)

y_pred = model.predict(x_test)                                      
print("acc_score : ", accuracy_score(y_test,y_pred))               

y_pred_best = model.best_estimator_.predict(x_test)                
print("최적 튠 ACC : ", accuracy_score(y_test,y_pred_best))        

'''
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[CV 1/3] END XGB__colsample_bylevel=0.7, XGB__colsample_bytree=0.9, XGB__eval_metric=error, XGB__learning_rate=0.01, XGB__max_depth=6, XGB__n_estimators=100, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.894 total time=19.8min
[CV 3/3] END XGB__colsample_bylevel=0.7, XGB__colsample_bytree=0.9, XGB__eval_metric=error, XGB__learning_rate=0.01, XGB__max_depth=6, XGB__n_estimators=100, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.898 total time=19.8min
[CV 2/3] END XGB__colsample_bylevel=0.7, XGB__colsample_bytree=0.9, XGB__eval_metric=error, XGB__learning_rate=0.01, XGB__max_depth=6, XGB__n_estimators=100, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.889 total time=19.9min
[CV 1/3] END XGB__colsample_bylevel=0.6, XGB__colsample_bytree=1, XGB__eval_metric=error, XGB__learning_rate=0.001, XGB__max_depth=5, XGB__n_estimators=200, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.854 total time=25.8min
[CV 3/3] END XGB__colsample_bylevel=0.6, XGB__colsample_bytree=1, XGB__eval_metric=error, XGB__learning_rate=0.001, XGB__max_depth=5, XGB__n_estimators=200, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.860 total time=25.9min
[CV 2/3] END XGB__colsample_bylevel=0.6, XGB__colsample_bytree=1, XGB__eval_metric=error, XGB__learning_rate=0.001, XGB__max_depth=5, XGB__n_estimators=200, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.849 total time=25.9min
[CV 2/3] END XGB__colsample_bylevel=0.6, XGB__colsample_bytree=0.6, XGB__eval_metric=error, XGB__learning_rate=0.1, XGB__max_depth=5, XGB__n_estimators=200, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.951 total time=26.5min
[CV 3/3] END XGB__colsample_bylevel=0.6, XGB__colsample_bytree=0.6, XGB__eval_metric=error, XGB__learning_rate=0.1, XGB__max_depth=5, XGB__n_estimators=200, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.954 total time=26.5min
[CV 1/3] END XGB__colsample_bylevel=0.6, XGB__colsample_bytree=0.6, XGB__eval_metric=error, XGB__learning_rate=0.1, XGB__max_depth=5, XGB__n_estimators=200, XGB__random_state=66, XGB__use_label_encoder=False;, score=0.954 total time=26.5min
'''