import numpy as np, pandas as pd, time,os,warnings,sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization

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

# print(datasets.info())
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1350 entries, 1 to 1350
Data columns (total 14 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   Overall Qual    1350 non-null   int64
 1   Gr Liv Area     1350 non-null   int64
 2   Exter Qual      1350 non-null   object     <-- object는 데이터형태의 최상위형태
 3   Garage Cars     1350 non-null   int64
 4   Garage Area     1350 non-null   int64
 5   Kitchen Qual    1350 non-null   object
 6   Total Bsmt SF   1350 non-null   int64
 7   1st Flr SF      1350 non-null   int64
 8   Bsmt Qual       1350 non-null   object
 9   Full Bath       1350 non-null   int64
 10  Year Built      1350 non-null   int64
 11  Year Remod/Add  1350 non-null   int64
 12  Garage Yr Blt   1350 non-null   int64
 13  target          1350 non-null   int64
dtypes: int64(11), object(3)
memory usage: 158.2+ KB
None
'''

# print(datasets.describe())        object형은 출력 안되어서 11개만나온다.
'''
       Overall Qual  Gr Liv Area  Garage Cars  Garage Area  ...   Year Built  Year Remod/Add  Garage Yr Blt         target
count   1350.000000  1350.000000  1350.000000  1350.000000  ...  1350.000000     1350.000000    1350.000000    1350.000000
mean       6.208889  1513.542222     1.870370   502.014815  ...  1972.987407     1985.099259    1978.471852  186406.312593
std        1.338015   487.523239     0.652483   191.389956  ...    29.307257       20.153244      25.377278   78435.424758
min        2.000000   480.000000     1.000000   100.000000  ...  1880.000000     1950.000000    1900.000000   12789.000000
25%        5.000000  1144.000000     1.000000   368.000000  ...  1955.000000     1968.000000    1961.000000  135000.000000
50%        6.000000  1445.500000     2.000000   484.000000  ...  1976.000000     1993.000000    1978.500000  165375.000000
75%        7.000000  1774.500000     2.000000   588.000000  ...  2002.000000     2004.000000    2002.000000  217875.000000
max       10.000000  4476.000000     5.000000  1488.000000  ...  2010.000000     2010.000000    2207.000000  745000.000000

[8 rows x 11 columns]
'''

# print(datasets.isnull().sum())    null값이 없다.
# print(datasets['Exter Qual'].value_counts())
# print(datasets['Kitchen Qual'].value_counts())
# print(datasets['Bsmt Qual'].value_counts())       <-- 이거 이미 내가 다 했던거. Project폴더의 housing에 보고 참고.

############################ 중복값 처리 !행기준! drop ###############################        
# print(f"제거전 : {train.shape}")
train = train.drop_duplicates().reset_index(drop=True)                                 # < -----------------------------         행 기준으로 중복값 찾아보고 제거.
# print(f"제거후 : {train.shape}")

############################ 이상치 확인 처리 ########################################
# outliers_loc = outliers(train['Garage Yr Blt'])
# print(outliers_loc)   # (array([254], dtype=int64),)  여기에 이상치가 있다고 말해주고 있다
# print(train.loc[[254],'Garage Yr Blt'])   # loc와 iloc정리하기
# Garage Yr Blt 2207년인것 삭제후 index번호 초기화
train = train.drop(train[train['Garage Yr Blt'] == 2207].index).reset_index(drop=True)


qual_cols = train.dtypes[train.dtypes == np.object].index       # <--------------------     이런방식으로 맵핑해 줄수도있다.
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
x_train_train,x_val,y_train_train,y_val = train_test_split(x_train,y_train,shuffle=True, random_state=66, train_size=0.8)

# print(x_train.shape,y_train.shape)  # (1079, 22) (1079,)
# print(x_test.shape,y_test.shape)    # (270, 22) (270,)

scaler_list = [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler(),QuantileTransformer(),
              PowerTransformer(method='yeo-johnson')] # PowerTransformer(method='box-cox')  # box-cox는 아마 error뜰것

    #  포문 사용할 경우.
for scaler in scaler_list:

  scaler = scaler

  x_train_train = scaler.fit_transform(x_train_train)
  x_test = scaler.transform(x_test)
  test = scaler.transform(test)

  ######## 베이지안 옵티마이제이션 써보자 #########
  params = {    # 실수형으로 파라미터 값을 받아서 연산처리한다.
          'max_depth' : (3, 7),
          'learning_rate' : (0.01, 0.2),
          'min_child_weight' : (0, 3),
          'subsample' : (0.5, 1),
          'colsample_bytree' : (0.2, 1),
          'reg_lambda' : (0.001, 10),   # 규제 2
          #'reg_alpha' : (0.01, 50),    # 규제 1
          #'gamma': (0,100)             성능차이가 딱히 없음
  }   
    
  bo = BayesianOptimization(f=xg_def,pbounds=params,random_state=66,verbose=2)

  bo.maximize(init_points=10, n_iter=200)   # 초기설정포인트 개수 몇가지 경우의 수로 스타트 할 것이냐?

  # print("============ bo.res ==============")
  # print(bo.res)                          # 모든 init_points + n_iter 경우의 수마다 어떤값이 들어갔는지 다 출력해준다.
  # print("============파라미터 튜닝 결과=====================")
  # print(bo.max)                            # best경우의 수를 출력해준다. 근데 target의 최대값을 best로 출력해준다.

  # 최소값을 보고싶다면 다른방법을써야한다.
  # bo.res의 모든 target값을 순.서대로 리스트에 담고 그 중에 최소값이 있는 위치의 index번호를 찾아내어 
  # 다시 bo.res에 접근하여 모든 리스트중에서 최소값이 있는 index번호의 값을 꺼내어 min_dict에 담아준다.
  target_list = []

  for result in bo.res:
      target = result['target']
      target_list.append(target)

  min_dict = bo.res[np.argmin(np.array(target_list))]
  print(f"{scaler}모델의 최소값 : {min_dict}")
  
'''
스케일러 안쓴 경우 상태 최소값 : {'target': 0.09381882007927493, 'params': {'colsample_bytree': 0.23946996029552725, 'learning_rate': 0.05620419620393713,
'max_depth': 3.58738891270194, 'min_child_weight': 0.36607189144595764, 'reg_lambda': 0.7487824644704899, 'subsample': 0.7067141248397533}}

StandardScaler()모델의 최소값 : {'target': 0.09366959028392545, 'params': {'colsample_bytree': 0.23946996029552725, 'learning_rate': 0.05620419620393713,
'max_depth': 3.58738891270194, 'min_child_weight': 0.36607189144595764, 'reg_lambda': 0.7487824644704899, 'subsample': 0.7067141248397533}}

MinMaxScaler()모델의 최소값 : {'target': 0.09400431470946025, 'params': {'colsample_bytree': 0.8970944438558832, 'learning_rate': 0.04121470647534586, 
'max_depth': 4.68575309305597, 'min_child_weight': 0.3373284963987425, 'reg_lambda': 8.909261111801708, 'subsample': 0.8469448925799528}}

MaxAbsScaler()모델의 최소값 : {'target': 0.09400431470946025, 'params': {'colsample_bytree': 0.8970944438558832, 'learning_rate': 0.04121470647534586, 
'max_depth': 4.68575309305597, 'min_child_weight': 0.3373284963987425, 'reg_lambda': 8.909261111801708, 'subsample': 0.8469448925799528}}

RobustScaler()모델의 최소값 : {'target': 0.09516356044924879, 'params': {'colsample_bytree': 0.5414997044907948, 'learning_rate': 0.04597330062303003, 
'max_depth': 3.3217367167342533, 'min_child_weight': 0.7447957859317819, 'reg_lambda': 0.19649115105950682, 'subsample': 0.7699147764843034}}

QuantileTransformer()모델의 최소값 : {'target': 0.09525317488555424, 'params': {'colsample_bytree': 0.22004175004873253, 'learning_rate': 0.04704296752856923, 
'max_depth': 6.8013800407563725, 'min_child_weight': 1.7585067217568227, 'reg_lambda': 0.7027655815789241, 'subsample': 0.5209820669014462}}   

PowerTransformer()모델의 최소값 : {'target': 0.09526880905879456, 'params': {'colsample_bytree': 0.8694452778558834, 'learning_rate': 0.11373676009655641, 
'max_depth': 3.168123952261128, 'min_child_weight': 2.3144362776000023, 'reg_lambda': 1.3585904116868197, 'subsample': 0.8285134833322685}} 


'''