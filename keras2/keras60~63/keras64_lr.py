x = 10
y = 10                      # 목표값
w = 0.5                     # 가중치 초기값
lr = 0.01                   
epochs = 300

for i in range(epochs):
    predict = x * w
    loss = (predict -y)**2
        
    up_predict = x * (w + lr)
    up_loss = (y - up_predict)**2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict)**2
    
    if up_loss > down_loss:
        w = w - lr
    else:
        w = w + lr 
        
    print(f"Step{i+1}\t Loss : {round(loss,4)}\t Predict : {round(predict,4)}\t Weight : {w}")
    
    
# tensor1이나 파이토치에서는 x에 weight를 곱한다.
# 행렬연산에서는 wx xw의 앞뒤 순서 바뀌면 차이가 크다.