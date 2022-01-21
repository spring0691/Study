import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

(x_train, _ ),(x_test, _ ) = mnist.load_data()    # 굳이 y쓰지도않을거기 때문에 _ 이렇게 표시해버린다.

real_x = np.append(x_train,x_test, axis=0)             # 이 쉬운걸 전에 몰랐어서 엄청 고생했었지 ㅋㅋㅋㅋ 
real_x = real_x.reshape(len(real_x),real_x.shape[1]*real_x.shape[2])
#print(real_x.shape)             # (70000,28,28)   ->  # (70000, 784)

###########################################
# 실습
# pca를 통해 0.95 이상인 n_components가 몇개?
###########################################
'''     -> 이건 값을 찾고 그 뒤에 fit을 돌려서 실제 정확도를 체크할때 하는거다.
useful_list = {}
for i in range(real_x.shape[1]):
    x = real_x
    pca = PCA(n_components=i+1)
    x = pca.fit_transform(x)
    
    pca_EVR = pca.explained_variance_ratio_
    pca = sum(pca_EVR)
    print(f'{i+1}일때 pca_EVR값{pca}')
    #print(f'{i+1}일때 누적합{np.cumsum(pca_EVR)}')
    
    if pca >= 0.95:
        useful_list[i+1] = pca 
'''

x = real_x
pca = PCA(n_components=784)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_     # n_components의 개수만큼 뭐가나옴 이걸 다 더하면?
cumsum = np.cumsum(pca_EVR)                 # 누적합을 구해줌 몇번째부터 일정 값 이상인지 확인가능

#print(cumsum)
print(np.argmax(cumsum>0.95) + 1)   # index번호 0부터 count하기 때문 154지점. 이 지점부터 0.95보다 커진다
print(np.argmax(cumsum>0.99) + 1)   # 331
print(np.argmax(cumsum>0.999) + 1)  # 486
print(np.argmax(cumsum) + 1)        # 713
# np.argmax -> [0.2 0.3 0.5] -> [0 0 1] -> 2로 반환해주는 원리 [0.2 0.5 0.3]이라면 1이나온다.
# 누적합 리스트는 당연히 뒤로갈수록 값이 커진다 -> 제일 뒤의 값을 1카운트하고 -> 그값을 인덱스 번호로 준다
# 누적합 리스트에서 (cumsum > 0.95) 이렇게 원하는 값의 지점을 찾고 그 지점을 넘어갈때 카운트를 세서
# 전체 리스트중 몇번째 위치부터 원하는 값이상으로 올라가는지 확인할 수 있다.
# argmax -> 최대값이 있는 위치를 반환해준다. 이 원리로 softmax를 되돌렸던거였다