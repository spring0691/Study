import numpy as np

aaa = np.array([[1,2,-20,4,5,6,7,8,30, 100, 500, 12, 13],
                [100,200,3,400,500,600,7,800,900,1000,1001,1002,99]])

aaa = np.transpose(aaa)     # (2,13) -> (13,2) ... 그냥 실무에서는 데이터제공받아서 열(col)별로 이상치 측정해야하니까


def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out,[25,50,75])     
    print("1사분위 : ", quantile_1)                                     
    print("q2 : ", q2)                                                  
    print("3사분위 : ", quantile_3)                                     
    iqr = quantile_3 - quantile_1   
    print("iqr : ", iqr)
    lower_bound = quantile_1 - (iqr * 1.5)     
    upper_bound = quantile_3 + (iqr * 1.5)      
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

# import matplotlib.pyplot as plt

# plt.boxplot(b,sym="bo")       
# plt.show()
