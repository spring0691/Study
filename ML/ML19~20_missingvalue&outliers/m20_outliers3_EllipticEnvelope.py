import numpy as np,pandas as pd

aaa = np.array([[1,2,-20,4,5,6,7,8,30, 100, 500, 12, 13]
                ,[100,200,3,400,500,600,7,800,900,1000,1001,1002,99]])

aaa = np.transpose(aaa)

a = []
b = []
for i in aaa:
    a.append(i[0])
    b.append(i[1])
a = (np.array(a)).reshape(-1,1)
b = (np.array(b)).reshape(-1,1)

from sklearn.covariance import EllipticEnvelope

# outlier를 찾아주는 함수
outliers = EllipticEnvelope(contamination=.4)  # contamination -> 오염
#contamination = 0~0.5 제일 끝에서부터 퍼센트를 지정해 그부분을 이상치로 정한다.
# 양쪽에서부터 잘라주기때문에 당연히 0.5가 최대치이다.
# outliers.fit(aaa)
# results = outliers.predict(aaa)
# print(results)
#[ 1  1  1  1  1  1  1  1  1 -1 -1  1  1]   이렇게 위치를 반환해준다.
print(type(a))
print(a.shape)
outliers.fit(b)
rs = outliers.predict(b)
print(rs)