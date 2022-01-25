import numpy as np

aaa = np.array([1,2,-20,4,5,6,7,8,30])#,12,13,500,100

def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out,[25,50,75])     # 사분위 나눠주고 중위값 구함
    print("1사분위 : ", quantile_1)                                     # 1사분위 값 역시 그 경계값의 평균
    print("q2 : ", q2)                                                  # 리스트가 짝수개일경우 중위값은 중위2개값의 평균
    print("3사분위 : ", quantile_3)                                     # 3사분위 값 역시 그 경계값의 평균
    iqr = quantile_3 - quantile_1   # 사분위 경계값을 뺀다.
    print("iqr : ", iqr)
    lower_bound = quantile_1 - (iqr * 1.5)      # 1사분위 - 중위값 * 1.5
    upper_bound = quantile_3 + (iqr * 1.5)      # 3사분위 + 중위값 * 1.5
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
    #   | 또는 이라는 의미. np.where 조건에 맞는값 찾기
    
# outliners_loc = outliers(aaa)
# print('이상치의 위치 : ', outliners_loc)

# 시각화 실습. boxplot로 그리기
import matplotlib.pyplot as plt

plt.boxplot(aaa,sym="bo")       # 데이터를 4분위 형식으로 그려준다. 가운데 박스가 중위값. 위 아래가 이상치이다.
plt.title('Box plot of aaa')    # 박스 위아래에 생긴 선이 1사분위 - iqr 값. 3사분위 + iqr값이다.
plt.xticks([1], ['aaa'])
plt.show()