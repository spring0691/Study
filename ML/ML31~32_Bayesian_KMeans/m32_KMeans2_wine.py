from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score,f1_score

datasets = load_wine()

wineDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

real_y = datasets.target

# print(np.unique(datasets.target,return_counts=True))    # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

i_list = [ ]
f1_list = [ ]

for i in range(100):
    
    kmeans = KMeans(n_clusters=3, random_state=i)     
    kmeans.fit(wineDF)

    pred_y = kmeans.labels_          
    
    f1 = f1_score(real_y,pred_y,average='macro')

    if f1 >= 0.70:
        f1_list.append(i)
        f1_list.append(f1)
        
print(i_list,f1_list)
