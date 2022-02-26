from tabnanny import verbose
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from catboost import CatBoostClassifier
import warnings, numpy as np, pandas as pd
from sklearn.feature_selection  import SelectFromModel

datasets = fetch_covtype()                  

x = datasets.data
x = pd.DataFrame(x, columns=datasets['feature_names'])

y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pf = PolynomialFeatures()
x_train = pf.fit_transform(x_train)
x_test = pf.transform(x_test)

model = CatBoostClassifier(task_type="GPU")

model.fit(x_train, y_train)

print(model.score(x_test,y_test))   

selection = SelectFromModel(model, threshold=0.2, prefit=True)   

select_x_train = selection.transform(x_train)  
select_x_test = selection.transform(x_test)  
print('drop 후 : ',select_x_train.shape,select_x_test.shape)

selection_model = CatBoostClassifier(task_type='GPU')

selection_model.fit(select_x_train,y_train)
    
print(f'model.score : {selection_model.score(select_x_test,y_test)}')

'''
fi = np.sort(model.feature_importances_).cumsum()
Fi = pd.DataFrame(model.feature_importances_.reshape(1,-1),columns=x.columns).sort_values(by=0,axis=1).cumsum(axis=1) 

del_num = np.argmax(Fi > 0.2)     
del_list = Fi.columns[:del_num]    

xx = x.drop(del_list,axis=1)
xx_train,xx_test = train_test_split(xx,train_size=0.8,shuffle=True, random_state=66)

model.fit(xx_train,y_train)
print(model.score(xx_test,y_test))   
'''