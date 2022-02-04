from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_boston,load_diabetes
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

datasets = load_diabetes()
x = datasets.data
y = datasets.target

choice = np.unique(y, return_counts=True)[1].min()

print(np.unique(y,return_counts=True))
print('\n\n',choice)
'''
if choice <= 3:
    print('나는야 회귀모델')
else : 
    print('나는야 분류모델')
'''