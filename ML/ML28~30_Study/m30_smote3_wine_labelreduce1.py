# 그냥 증폭해서 성능바교

from tabnanny import verbose
import numpy as np, pandas as pd, warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV         #,cross_val_score,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.metrics import f1_score,accuracy_score
from sklearn.feature_selection  import SelectFromModel
from imblearn.over_sampling import SMOTE

warnings.filterwarnings(action='ignore')

#1. 데이터

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)
x = datasets.drop(['quality'],axis=1)   # (4898, 11)    
y = datasets['quality']  

y = np.where(y<=5,'Good',np.where(y==6,'Normal',np.where(y<=9,'Bad',y)))

le = LabelEncoder()
y = le.fit_transform(y)
# print(np.unique(y,return_counts=True)) [0, 1, 2] [1060, 1640, 2198]


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y)  
# print(np.unique(y_train,return_counts=True)) #[0, 1, 2]    [ 848, 1312, 1758]


# train데이터만 증폭시켜보자
smote = SMOTE(random_state=66,k_neighbors=2)    # n_neighbors = k_neighbors + 1 인접한값을 참조하여 증폭하기때문에 최대값이 정해져있다.
x_train,y_train = smote.fit_resample(x_train,y_train) 
# print(np.unique(y_train,return_counts=True))  #[0, 1, 2] [1758, 1758, 1758]


x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66,stratify=y_train)  
# print(np.unique(y_train,return_counts=True))  #[0, 1, 2] [1406, 1407, 1406]


model = XGBClassifier(
    n_estimators = 1000,
    max_depth = 9,
    learning_rate = 0.05,
    reg_lambda = 1,
    reg_alpha = 1,
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor',
    eval_metric='merror',
    use_label_encoder=False,
)
model.fit(x_train,y_train,verbose=True,early_stopping_rounds=100,eval_set=[(x_val,y_val)])

# kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=66)
# cross_val_score(model,x_train,y_train,cv=kfold,verbose=2,early_stopping_rounds=100)

print(f"md.score : {model.score(x_test,y_test)}")                             #0.689795918367347
print(f"ac_score : {accuracy_score(y_test,model.predict(x_test))}")             #0.689795918367347
print(f"f1_score : {f1_score(y_test,model.predict(x_test),average='macro')}")   #0.6935184759783691
print(f"f1_score : {f1_score(y_test,model.predict(x_test),average='micro')}")   #0.689795918367347
