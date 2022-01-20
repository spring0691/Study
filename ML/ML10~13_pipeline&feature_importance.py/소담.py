import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer, fetch_covtype

datasets = {'Iris':load_iris(),
            'Wine':load_wine(),
            'Diabets':load_diabetes(),
            'Cancer':load_breast_cancer(),
            'Boston':load_boston(),
            'FetchCov':fetch_covtype(),
            #'Kaggle_Bike':'Kaggle_Bike'
            }

model_1 = DecisionTreeClassifier(random_state=66, max_depth=5)
model_1r = DecisionTreeRegressor(random_state=66, max_depth=5)

model_2 = RandomForestClassifier(random_state=66, max_depth=5)
model_2r = RandomForestRegressor(random_state=66, max_depth=5)

model_3 = XGBClassifier(random_state=66)
model_3r = XGBRegressor(random_state=66)

model_4 = GradientBoostingClassifier(random_state=66)
model_4r = GradientBoostingRegressor(random_state=66)

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

#path = "D:\\Study\\_data\\bike\\"
#train = pd.read_csv(path + "train.csv")

model_list = [model_1,model_2,model_3,model_4]
model_list_r = [model_1r,model_2r,model_3r,model_4r]

model_name = ['DecisionTree','RandomForest','XGB','GradientBoosting']

for (dataset_name, dataset) in datasets.items():
    print(f'------------{dataset_name}-----------')
    print('====================================')    
    
    # if dataset_name == 'Kaggle_Bike':
    #     y = train['count']
    #     x = train.drop(['casual', 'registered', 'count'], axis=1)        
    #     x['datetime'] = pd.to_datetime(x['datetime'])
    #     x['year'] = x['datetime'].dt.year
    #     x['month'] = x['datetime'].dt.month
    #     x['day'] = x['datetime'].dt.day
    #     x['hour'] = x['datetime'].dt.hour
    #     x = x.drop('datetime', axis=1)
    #     y = np.log1p(y)        
    
    x = dataset.data
    y = dataset.target    

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=0.8, shuffle=True, random_state=66)
    plt.figure(figsize=(15,10))
    for i in range(4):        
        plt.subplot(2, 2, i+1)               # nrows=2, ncols=1, index=1
        if dataset_name == 'Cancer':
            model_list_r[i].fit(x_train, y_train)
            score = model_list_r[i].score(x_test, y_test)
            feature_importances_ = model_list_r[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list_r[i])    
            
        else: 
            model_list[i].fit(x_train, y_train)
            score = model_list[i].score(x_test, y_test)
            feature_importances_ = model_list[i].feature_importances_

            # y_predict = model_list[i].predict(x_test)
            # acc = accuracy_score(y_test, y_predict)
            print("score", score)
            # print("accuracy_score", acc)
            print("feature_importances_", feature_importances_)
            plot_feature_importances_dataset(model_list[i])    
            plt.ylabel(model_name[i])
            plt.title(dataset_name)

    plt.tight_layout()
    plt.show()