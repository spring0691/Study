import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     
import torch.nn.functional as F 
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터 정제해서 값 도출
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])   
y = np.array([11,12,13,14,15,16,17,18,19,20])

x = np.transpose(x) # (2,10) -> (10,2)

x = torch.FloatTensor(x).to(DEVICE)   
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   # (10,) -> (10,1)

# print(x.shape,y.shape)  # torch.Size([10, 2]) torch.Size([10, 1])

#2. 모델구성
model = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,1),
).to(DEVICE)    #Linear을 Sequential로 묶었다.

#3. 컴파일, 훈련     
criterion = nn.MSELoss()    
optimizer = optim.Adam(model.parameters(), lr=0.01) 

def train(model, criterion, optimizer, x, y):
    #model.train()   # 훈련모드 default로 들어가 있음.
    optimizer.zero_grad()   
    
    hypothesis = model(x)  
    
    loss = criterion(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis,y)   # 이렇게 선언하면 돌아간다.
    # loss = F.mse_loss(hypothesis,y)     # 이것도 잘 돌아간다.
    loss.backward()    
    optimizer.step()    
    return loss.item()  

epochs = 0
while True:
    epochs += 1
    loss = train(model,criterion,optimizer,x,y)    # 현재는 data loader 쓰지 않는 상태
    print(f'epoch : {epochs}, loss : {loss}')
    # time.sleep(0.1)
    if loss < 0.000001: break
print('==================================================================================')

#4. 평가, 예측
def evaluate(model,criterion, x, y):
    model.eval()        # torch는 eval로 평가한다. 평가모드

    with torch.no_grad():   # gradient를 갱신하지않겠다.
        predict = model(x)
        loss2 = criterion(predict, y)
        # loss2 = nn.MSELoss()(predict,y)   

    return loss2.item()

loss2 = evaluate(model, criterion,x, y)
print(f'최종 loss : {loss2}')

result = model(torch.Tensor([[10,1.3]]).to(DEVICE))  
print(f'[10,1.3]의 예측값 : {result.item()}')
'''
최종 loss : 9.845498425420374e-07
4의 예측값 : 19.99825668334961
'''