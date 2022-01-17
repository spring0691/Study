import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()
features = datasets.data    # x
label = datasets.target     # y

model = LogisticRegression(max_iter=5000)

result_skfold = StratifiedKFold(n_splits=4) # 분류모델
idx_iter=0
cv_accuracy = []                            # n_splits의 값을 담을 리스트 


# StratifiedKFold의 split() 호출시 반드시 레이블 데이터 셋도 추가 입력 필요
for train_index, test_index in result_skfold.split(features, label) :
    
    # split()으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    # 학습 및 예측
    allAlgorithms = all_estimators(type_filter = 'classifier') 
    for (name, algorithm) in allAlgorithms:  
        try:
            model = algorithm()
            model.fit(x_train, y_train)
            
            pred = model.predict(x_test)
            acc = accuracy_score(y_test, pred)
            print(name, '의 정답률: ', acc)
        except:  
            print(name, "예외(에러남)")

    # 반복 시 마다 정확도 측정
    idx_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'.format(idx_iter, accuracy, train_size, test_size))  
    print('#{0} 검증 세트 인덱스 : {1}'.format(idx_iter, test_index))
    cv_accuracy.append(accuracy)
    

# 교차 검증별 정확도 및 평균 정확도 계산
print('\n교차 검증별 정확도 : ', np.round(cv_accuracy, 4))
print('평균 검증 정확도 : ', np.mean(cv_accuracy))