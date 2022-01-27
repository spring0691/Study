import numpy as np,pandas as pd

# def Nan_changes(data):
#     quantile_1, q2, quantile_3 = np.percentile(data,[25,50,75])     
#     iqr = quantile_3 - quantile_1   
#     lower_bound = quantile_1 - (iqr * 1.5)      
#     upper_bound = quantile_3 + (iqr * 1.5)   
#     # del_index = np.where((data>upper_bound) | (data<lower_bound))
#     # data[11] = pd.DataFrame(data).replace(data[11],np.NaN)
#     data = data[(lower_bound < data) & (data < upper_bound)]
#     return data

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

aaa = np.sort(np.array([1,2,-20,4,5,6,7,8,30,40,60,1000,200]))
aaa = pd.DataFrame(aaa)

print(remove_outlier(aaa))
