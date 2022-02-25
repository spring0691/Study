import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)

print(x)
print(x.shape)

pf = PolynomialFeatures(degree=2)
# degree의 차수만큼 칼럼을 생성하여 데이터를 증폭시킨다.
# 기존 컬럼 x1,x2 -> x1,x1^2,x1*x2,x2^2,x2 해서 칼럼이 5개로 늘어나고 + 앞에 1이들어간 칼럼하나 추가시켜서 총 6개가 된다.
xp = pf.fit_transform(x)
print(xp,'\n',xp.shape)

# 이걸 이용해서 칼럼 증폭 시키고 -> Feature_importances 돌려서 칼럼 drop시킨 후 돌리면 성능 향상된다.
##############################################################################################################

x = np.arange(12).reshape(4,3)
print(x)
print(x.shape)

pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(f'{xp}\n{xp.shape}')          # (4, 10)
##############################################################################################################

x = np.arange(8).reshape(4,2)
print(x)
print(x.shape)

pf = PolynomialFeatures(degree=3)
xp = pf.fit_transform(x)
print(f'{xp}\n{xp.shape}')          # (4, 10)   1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3