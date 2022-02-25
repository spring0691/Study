from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings, numpy as np

warnings.filterwarnings(action='ignore')

# datasets = load_boston()                    # (506, 13) (506,)
datasets = fetch_california_housing()       # (20640, 8) (20640,)

x = datasets.data
y = datasets.target

# print(datasets.feature_names)
# print(datasets.DESCR)           # pandas에선 describe()해주면 정보나옴.
# print(x.shape,y.shape)

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())




























exit()
# model.fit(x_train, y_train)

# print(model.score(x_test,y_test))   # 0.7795056314949791
# print(model.score(x_test,y_test))   # 0.77950563149498      standard 적용.

from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')   # scoring는 평가지표, sklearn.metrics.SCORERS.keys()로 확인가능.
# model.fit(x_train, y_train)
# print(np.mean(scores))
# print(model.score(x_test,y_test))

from sklearn.preprocessing import PolynomialFeatures#,PowerTransformer
pf = PolynomialFeatures(degree=2)   # data형태가 2차함수처럼 값이 휘어져있을때 칼럼을 증폭시켜서 1차함수형태로 만들어 연산성능을 높여준다
xp = pf.fit_transform(x)
# print(xp.shape)                   # 칼럼이 엄청 늘어났다.
x_train, x_test, y_train, y_test = train_test_split(xp, y, test_size=0.1, random_state=66)

# scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')   # scoring는 평가지표, sklearn.metrics.SCORERS.keys()로 확인가능.
model.fit(x_train, y_train)
# print(np.mean(scores))
print(model.score(x_test,y_test))
