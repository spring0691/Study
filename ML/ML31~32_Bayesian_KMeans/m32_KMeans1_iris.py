from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_iris()

irisDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

kmeans = KMeans(n_clusters=6, random_state=66)      # random_state가 n_clusters의 좌표상에서의 스타팅 포인트를 잡아준다.
kmeans.fit(irisDF)
# kmeans.labels_     # 결과물
print(kmeans.labels_)

# irisDF['cluster'] = kmeans.labels_          # 이런식으로 새로운 칼럼을 생성과 동시에 붙여줄 수 있다.
# irisDF['target'] = datasets.target

pred_y = kmeans.labels_          
real_y = datasets.target

# print(accuracy_score(real_y,pred_y))
# print(accuracy_score(real_y, np.sort(kmeans.labels_)))

# print(real_y)
