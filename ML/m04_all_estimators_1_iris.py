from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings(action='ignore')
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

allAlgorithms = all_estimators(type_filter='classifier')  # 모든 분류모델들 하나로 묶어줌 짱짱 와우~    41개
#allAlgorithms = all_estimators(type_filter='regressor')    # 모든 회귀모델들 하나로 묶어줌 짱짱 와우~  55개
#print('allAlgorithms : ',allAlgorithms)        모델과 위치가 들어있는 딕셔너리들이 리스트에 들어있다
#print('모델의 갯수 : ', len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        print(name,'에서 오류떴어~')
   
