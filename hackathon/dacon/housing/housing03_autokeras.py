import numpy as np, pandas as pd, time,os,warnings,sys
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization
import autokeras as ak                          ###############!!!!!!!!!!!!!!!!!!! autokeras는 tensorflow다 머신러닝이 아니다.

warnings.filterwarnings(action='ignore')
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 170)

def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out,[25,50,75])                                      
    iqr = quantile_3 - quantile_1  
    lower_bound = quantile_1 - (iqr * 1.5)      
    upper_bound = quantile_3 + (iqr * 1.5)     
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
  
def xg_def(max_depth, learning_rate, min_child_weight, 
              subsample, colsample_bytree,reg_lambda):
    xg_model = XGBRegressor(
      max_depth = int(max_depth),
      learning_rate = learning_rate,
      n_estimators = 3000,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      reg_lambda = reg_lambda
    )
    xg_model.fit(x_train_train,y_train_train,eval_set=[(x_val,y_val)],eval_metric='mae',verbose=False,early_stopping_rounds=100)
    y_predict = xg_model.predict(x_test)
    
    nmae = NMAE(np.round(np.expm1(y_test)),np.round(np.expm1(y_predict)))
    return nmae
  
path = os.path.dirname(os.path.realpath(__file__)) + '/'
train = pd.read_csv(path + 'data/train.csv').drop(['id'],axis=1)
test = pd.read_csv(path + 'data/test.csv').drop(['id'],axis=1)
submit_sets = pd.read_csv(path + 'data/sample_submission.csv', index_col=0, header=0)

train = train.drop_duplicates().reset_index(drop=True)                                
train = train.drop(train[train['Garage Yr Blt'] == 2207].index).reset_index(drop=True)

qual_cols = train.dtypes[train.dtypes == np.object].index      
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

train = label_encoder(train, qual_cols)
test = label_encoder(test, qual_cols)

################################### 분류형 컬럼을 one hot endocing #########################################
train = pd.get_dummies(train, columns=['Exter Qual','Kitchen Qual','Bsmt Qual'])
test = pd.get_dummies(test, columns=['Exter Qual','Kitchen Qual','Bsmt Qual'])


# x,y정의 후 y값에 로그(선택)
x = train.drop(['target'],axis=1)
y = train['target']
y = np.log1p(y)

test = test.values  # numpy로 변경

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)
# x_train_train,x_val,y_train_train,y_val = train_test_split(x_train,y_train,shuffle=True, random_state=66, train_size=0.8)

ak_model = ak.StructuredDataRegressor(
    overwrite=True, max_trials=2
)

start = time.time()
ak_model.fit(x_train,y_train, epochs=10,validation_split=0.2)  # epoch는 1000번 validation_split는 0.2 early stop은 50번이 default
end = time.time()-start

model = ak_model.export_model()         # 모델에서 50번 trials한 결과중 제일 좋은 모델을 추출하겠다.

#save_model로 저장한다. 

y_predict = ak_model.predict(x_test).reshape(-1)
# print(y_predict.shape)

results = model.evaluate(x_test,y_test)
# print(results)

nmae = NMAE(np.round(np.expm1(y_test)).astype(int), np.round(np.expm1(y_predict)).astype(int))
# print(nmae)

############################# 파일 제출 설정 ################################
colsample_bytree = 0.8819, 
learning_rate = 0.119, 
max_depth = 5, 
min_child_weight = 1.7166, 
n_estimators = 7421, 
reg_lambda = 8.355, 
subsample = 0.905


nowtime = datetime.now()
now_date = nowtime.strftime("%m%d_%H%M")

y_submit = model.predict(test)
y_submit = np.expm1(y_submit)
submit_sets.target = y_submit
submit_sets.to_csv(f"{path}submit/{now_date}_{round(nmae,4)}.csv",index=False)

model.summary()     # summary또한 확인 할 수 있다.  tensorflow 모델이기때문에 가능
'''
Model: "model_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_4 (InputLayer)        [(None, 22)]              0

 multi_category_encoding_3 (  (None, 22)               0
 MultiCategoryEncoding)

 normalization_2 (Normalizat  (None, 22)               45
 ion)

 dense_6 (Dense)             (None, 32)                736

 re_lu_6 (ReLU)              (None, 32)                0

 dense_7 (Dense)             (None, 32)                1056

 re_lu_7 (ReLU)              (None, 32)                0

 regression_head_1 (Dense)   (None, 1)                 33

=================================================================
Total params: 1,870
Trainable params: 1,825
Non-trainable params: 45
_________________________________________________________________
'''


with open(f"{path}submit/{now_date}_{round(nmae,4)}submit.txt", "a") as file:
    file.write("\n==============================")
    file.write(f'저장시간 : {now_date}\n')
    file.write(f'scaler : 안썼어용\n')
    file.write(f"colsample_bytree : {colsample_bytree}\n")
    file.write(f"learning_rate : {learning_rate}\n")
    file.write(f"max_depth : {max_depth}\n")
    file.write(f"min_child_weight : {min_child_weight}\n")
    file.write(f"n_estimators : {n_estimators}\n")
    file.write(f"reg_lambda : {reg_lambda}\n")
    file.write(f"subsample : {subsample}\n")
    file.write(f"걸린시간 : {round(end,4)}\n")
    file.write(f"evaluate : {results}\n")
    file.write(f"NMAE : {round(nmae,4)}")
    # file.close()  with로 열여줘서 안닫아줘도 됌.