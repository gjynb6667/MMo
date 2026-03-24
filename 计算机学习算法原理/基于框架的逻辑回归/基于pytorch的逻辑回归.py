import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1.散点输入
class1_points = np.array([[1.9, 1.2],
                          [1.5, 2.1],
                          [1.9, 0.5],
                          [1.5, 0.9],
                          [0.9, 1.2],
                          [1.1, 1.7],
                          [1.4, 1.1]])

class2_points = np.array([[3.2, 3.2],
                          [3.7, 2.9],
                          [3.2, 2.6],
                          [1.7, 3.3],
                          [3.4, 2.6],
                          [4.1, 2.3],
                          [3.0, 2.9]])
x_data = np.concatenate(class1_points,class2_points)
y_data = np.concatenate(np.zeros(len(class1_points)),np.ones(len(class2_points)))
torch.manual_seed(1)
def sigmoid(x):
    return 1/1+np.exp(-x)
class linear(nn.Module):
    def __init__(self):
        super(linear,self).__init__()
        self.fc = nn.linear(2,1)
    def forward(self,x):
        x = self.fc(x)
        return torch.sigmoid(x)
model = linear()
cri = torch.nn.BCELoss()
optimzier = torch.optim.SGD(model.parameters(),lr = 0.05)
epochs = 1000
for epoc in range(1,epochs+1):
    input = torch.tensor(x_data,dtype=torch.float32)
    label = torch.tensor(y_data,dtype=torch.float32)
    outputs = model(input)
    loss = cri(outputs,label)
    optimzier.zero_grad()
    loss.backward()
    optimzier.step()