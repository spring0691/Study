import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer,fetch_covtype
from sklearn.decomposition import PCA   # decomposition 분해
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    #이게 LDA   
# import sklearn as sk
# print(sk.__version__)

#1. 데이터로드 및 정제

# datasets = load_boston()
#datasets = fetch_california_housing()
# datasets = load_breast_cancer()
datasets = fetch_covtype()

x = datasets.data       
y = datasets.target
#print(x.shape)
#print(datasets.feature_names)

# from sklearn.datasets import fetch_openml
# housing = fetch_openml(name="house_prices", as_frame=True)

# pca = PCA(n_components=13)   
lda = LinearDiscriminantAnalysis()   # n_components의 개수는 features의 최소값 또는 label의 개수 -1 보다 클 수 없다
# x = pca.fit_transform(x)
x = lda.fit_transform(x, y)         # y를 보고 한다는게 어마어마한거다. 어마어마한 압축률과 성능을 자랑한다.
# print(x.shape)                      # default값은 y_classes - 1 값이다.
# print(x[:5])

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=66, shuffle=True)

#2. 모델
from xgboost import XGBRegressor,XGBClassifier
model = XGBClassifier(eval_metric='error')
# model = XGBRegressor

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
results = model.score(x_test,y_test)
print('결과 : ', results)

# breast_cancer
# xgboost default
# 결과 :  0.9736842105263158

# LDA 
# 결과 :  0.9824561403508771

# fetch_covtype
# LDA
# 결과 :  0.7882498730669604