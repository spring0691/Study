import pandas as pd, numpy as np,sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# 출력 관련 옵션
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_row',100)
pd.set_option('display.max_columns',50)
pd.set_option('display.width', 150)

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)  # (4898, 12) index와 header은 default값.
# print(datasets.describe())
'''
       fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density  \
count    4898.000000       4898.000000  4898.000000     4898.000000  4898.000000          4898.000000           4898.000000  4898.000000
mean        6.854788          0.278241     0.334192        6.391415     0.045772            35.308085            138.360657     0.994027
std         0.843868          0.100795     0.121020        5.072058     0.021848            17.007137             42.498065     0.002991
min         3.800000          0.080000     0.000000        0.600000     0.009000             2.000000              9.000000     0.987110
25%         6.300000          0.210000     0.270000        1.700000     0.036000            23.000000            108.000000     0.991723
50%         6.800000          0.260000     0.320000        5.200000     0.043000            34.000000            134.000000     0.993740
75%         7.300000          0.320000     0.390000        9.900000     0.050000            46.000000            167.000000     0.996100
max        14.200000          1.100000     1.660000       65.800000     0.346000           289.000000            440.000000     1.038980

                pH    sulphates      alcohol      quality
count  4898.000000  4898.000000  4898.000000  4898.000000
mean      3.188267     0.489847    10.514267     5.877909
std       0.151001     0.114126     1.230621     0.885639
min       2.720000     0.220000     8.000000     3.000000
25%       3.090000     0.410000     9.500000     5.000000
50%       3.180000     0.470000    10.400000     6.000000
75%       3.280000     0.550000    11.400000     6.000000
max       3.820000     1.080000    14.200000     9.000000
'''
# print(datasets.info())            # 너무나 깔끔하게 정보를 보여준다. 결측치도 있나없나 보여준다.
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4898 entries, 0 to 4897
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   fixed acidity         4898 non-null   float64
 1   volatile acidity      4898 non-null   float64
 2   citric acid           4898 non-null   float64
 3   residual sugar        4898 non-null   float64
 4   chlorides             4898 non-null   float64
 5   free sulfur dioxide   4898 non-null   float64
 6   total sulfur dioxide  4898 non-null   float64
 7   density               4898 non-null   float64
 8   pH                    4898 non-null   float64
 9   sulphates             4898 non-null   float64
 10  alcohol               4898 non-null   float64
 11  quality               4898 non-null   int64
dtypes: float64(11), int64(1)
memory usage: 459.3 KB
None
'''
# 이제부터는 결측치, 이상치 확인하는 작업도 꼭. 하고 가자!!
# isnull()로 nan값 확인. isnull().sum()하면 다 더해서 깔끔하게 보여준다.
# fixed acidity           0
# volatile acidity        0
# citric acid             0
# residual sugar          0
# chlorides               0
# free sulfur dioxide     0
# total sulfur dioxide    0
# density                 0
# pH                      0
# sulphates               0
# alcohol                 0
# dtype: int64                  결측치는 없다!! 여기서 말하는 int는 저기나오는 0의 dtype이 int란 뜻이다 혼동하지 말자.

datasets = datasets.to_numpy()
# print(type(datasets))   # <class 'numpy.ndarray'>
# print(datasets.shape)   # (4898, 12)

x = datasets[:, :11]    #4898, 11)
y = datasets[:, 11]     #(4898,)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y) # yes의 y가 아니라. y의 y다. 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_jobs=-1)

#3. 훈련
model.fit(x_train,y_train,eval_metric='merror')

#4. 평가, 예측
score = model.score(x_test,y_test)
print(f'model.score : {score}')

y_predict = model.predict(x_test)

acc = accuracy_score(y_test,y_predict)
f1= f1_score(y_test,y_predict,average='macro')  # [‘micro’, ‘macro’, ‘samples’,’weighted’ 중 하나 선택]
print(f'acc_score : {acc}')
print(f'f1_score : {f1}')

# f1_score는 원래 2진분류에서 데이터가 불균형일 경우 positive, negative로 이진 분류해야 precision, recall, f1을 구할 수 있으니 
# 다중 클래스 분류에서는 f1 스코어를 사용 할 수 없다.
'''
'micro':

Calculate metrics globally by counting the total true positives, false negatives and false positives.

true positive와 false negative, false positive의 합을 산출해 스코어를 계산한다.

-> 각 샘플이나 예측에 동일한 가중치를 부여하고자 할 때 사용한다.

'macro':

Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

각 레이블의 unweighted된 평균(레이블이 불균형한 멀티-클래스 분류 문제에서)을 계산한다. 레이블이 불균형을 따로 고려하지 않는다.

모든 클래스에 동일한 가중치를 부여하여 분류기의 전반적인 성능을 평가한다. 가장 빈도 높은 클래스 레이블의 성능이 중요하다.

'weighted':

Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

각 레이블이 불균형해도, weight를 주어 평가지표를 계산한다. precision과 recall의 합이 아닌 F-score를 야기할 수 있다.

'samples':

Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score

각 레이블의 평가지표를 평균 낸다.(accuracy_score와 다른 멀티-레이블 분류 문제에서만 의미 있다.)

고로 average='sample', 아니면 'weighted', 아니면 'micro'를 사용해도 괜찮을 것 같다.

-> 하지만 모델에 넣어봤을 때, 'sample'의 경우 동작하지 않았고 'weighted'를 넣으니 동작했다.
'''





'''
x = datasets.drop(['quality'],axis=1)   # (4898, 11)    
y = datasets['quality']                 # [3, 4, 5, 6, 7, 8, 9]가 각각 [  20,  163, 1457, 2198,  880,  175,    5] 되어있다.

le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66,stratify=y)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66, stratify=y_train)

model = XGBClassifier(
    objective = 'multi : softmax',
    num_class = -1,
    use_label_encoder=False,
    n_estimators = 1000,     
    learning_rate = 0.005,
    max_depth = 20,          
    min_child_weight = 1,
    subsample=1,
    colsample_bytree = 1,
    reg_alpha = 0,          # 규제  L1       -> 둘 중에 하나만 할수도 있다.
    reg_lambda = 1,          # 규제  L2      -> 응용해서 나온개념 릿지와 랏소   가중치 규제하는것.
    tree_method = 'gpu_hist',
    predictor = 'gpu_predictor'
)

model.fit(x_train,y_train,eval_metric='merror',eval_set=[(x_train,y_train),(x_val,y_val)],early_stopping_rounds=50)

print(model.score(x_test,y_test))
'''