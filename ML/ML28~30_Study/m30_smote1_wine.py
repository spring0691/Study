import numpy as np, pandas as pd
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score

datasets = load_wine()
x = datasets.data
y = datasets.target
# print(x.shape,y.shape)              # (178, 13) (178,)
# print(pd.Series(y).value_counts())  # 1    71   0    59 2    48
# DataFarme은 행렬 Series는 벡터

x_new = x[:-30]
y_new = y[:-30]
# print(np.unique(y_new,return_counts=True))  # (array([0, 1, 2]), array([59, 71, 18], dtype=int64)) 2번 개수가 30개 줄어들었다

x_train,x_test,y_train,y_test = train_test_split(x_new,y_new,shuffle=True, random_state=66, train_size=0.75,stratify=y_new)

model = XGBClassifier(n_jobs=4,eval_metric='merror',use_label_encoder=False)
model.fit(x_train,y_train)

score = round(model.score(x_test,y_test),4)
print(f"model.score : {score}")

y_predict = model.predict(x_test)
acc = round(accuracy_score(y_test,y_predict),4)
print(f"  acc_score : {acc}")

# model.score 0.9778 -> 0.9459      y에서 2번 라벨의 개수가 30개 줄어드니까 정확도가 떨어졌다. 
# 라벨이 골고루 분포되어있을때 값이 잘 나온다.  -> 불균형할 경우 직접 증폭시켜주는게 좋다.

print("======================== SMOTE 적용 ============================")

smote = SMOTE(random_state=66)
x_train,y_train = smote.fit_resample(x_train,y_train)         
# train은 균형하게 분포시키고 test에서 불균형한 원본 그대로 던져주면 성능을 볼수 있다.
# print(np.unique(y_train,return_counts=True))   # (array([0, 1, 2]), array([44, 53, 14], dtype=int64))
# print(np.unique(y_train,return_counts=True))    # (array([0, 1, 2]), array([53, 53, 53], dtype=int64))
# 불균형했던 분포를 증폭시켜서 균형한 분포로 바꿔준다.

model.fit(x_train,y_train)

score = round(model.score(x_test,y_test),4)
print(f"model.score : {score}")

y_predict = model.predict(x_test)
acc = round(accuracy_score(y_test,y_predict),4)
print(f"  acc_score : {acc}")

# 0.9459 -> 0.973   확실하게 성능이올라갔다. 