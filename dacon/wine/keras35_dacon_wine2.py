import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
path = "../_data/dacon/wine/"

train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

x = train.drop(columns=['id', 'quality','pH'], axis=1)  #'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide','total sulfur dioxide', 'pH', 'sulphates'
y = train['quality']
test_file = test_file.drop(columns=['pH','id'], axis=1)

x.type = LabelEncoder().fit_transform(x.type) 
test_file.type = LabelEncoder().fit_transform(test_file.type)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=90)

scaler = MinMaxScaler()      #MaxAbsScaler()RobustScaler()StandardScaler()

# # cnn방식 scaler    
# # x_train = scaler.fit_transform(x_train).reshape(len(x_train),5,2,1)
# # x_test = scaler.transform(x_test).reshape(len(x_test),5,2,1)  
# # test_file = scaler.transform(test_file).reshape(len(test_file),5,2,1)

# # dnn방식 scaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  
test_file = scaler.transform(test_file)

rfc = RandomForestClassifier(n_estimators=100, max_depth=100,random_state=7)
#vc  = VotingClassifier(estimators=100)
#kc  = KNeighborsClassifier(estimators=100, max_depth=300,random_state=5)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

rfc.fit(x_train, y_train)

predict = rfc.predict(x_test)

print(accuracy_score(y_test , predict))

acc = str(round(accuracy_score(y_test , predict),4))

#rfc.save(f"./_save/keras32_8_wine{acc}.h5")
joblib.dump(rfc, f"./_save/keras32_8_wine{acc}.joblib") 
results = rfc.predict(test_file)
submit_file['quality'] = results
submit_file.to_csv(path+f"result/acc_{acc}.csv", index = False)