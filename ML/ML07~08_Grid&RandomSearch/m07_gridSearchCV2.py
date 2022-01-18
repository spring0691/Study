from itertools import repeat
from tabnanny import verbose
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


#2. 모델구성

#model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True)   'C': 1, 'degree': 3, 'gamma': 0.01, 'kernel': 'linear'
model = SVC(C=1,kernel='linear',gamma = 0.01, degree=3)

#3. 훈련

model.fit(x_train,y_train)

#4. 평가,예측

print("model.score : ", model.score(x_test,y_test))    

y_pred = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test,y_pred))

