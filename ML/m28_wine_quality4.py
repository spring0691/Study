from itertools import groupby
import pandas as pd, numpy as np,sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,PowerTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

# 출력 관련 옵션
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',100)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 150)

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)  # (4898, 12) index와 header은 default값.

# datasets = datasets.to_numpy()
# x = datasets[:, :11]    #(4898, 11)
# y = datasets[:, 11]     #(4898,)

################# 그래프 그려봐!!! ###################

# pd.value_counts를 그래프로 구현하기. 단 value_counts사용금지
# print(list[str(datasets['quality'].value_counts())])    # [6    2198\5    1457\7     880\8     175\4     163\3      20\9       5]
# print(datasets.value_counts())    datasets전체에 value_counts를 하면 특정조건의 품질이 몇 종류가 몇개씩 있나 알려준다.    

### groupby 와 count() 이용해서 그리기 기준 quality. plt.bar이용 

# groubby는 통계자료에서 굉장히 많이 사용한다. 그룹핑 시켜서 데이터의 합을 구하거나 평균치를 구하거나... 등등

# test = datasets.groupby(datasets['quality'])
# groubby 함수로 데이터를 그룹핑하면 DataFrameGroupBy 객체가 리턴된다. 이 상태로는 아무것도 못한다
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001DF0A0C6F70>

# groubby 함수로 그룹핑을 했으면 반드시 통계함수를 적용시켜야 한다.
# 둘의 차이가 무엇일까??? 
# datasets.groupby(['quality']).size().plot(kind='bar', rot=0)
# datasets.groupby(['quality']).count().plot(kind='bar', rot=0)
# plt.show()

# 선생님방식
# datasets.groupby('quality')['quality'].count().plot(kind='bar',rot=0)
# plt.show()
######################################################

