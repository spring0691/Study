# 실습

# 모델 : RandomForestClassifier 
parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs : ' : [-1, 2, 4, 6]}
]