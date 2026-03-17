import  torch
import numpy as np
from matplotlib import gridspec
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
from torchsummary import summary
# 1、散点输入，定义输入数据
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
data = np.array(data)
x_data = data[:,0]
y_data = data[:,1]
x_torch = torch.tensor(x_data,dtype=torch.float32)
print(x_torch.shape)
y_torch = torch.tensor(y_data,dtype=torch.float32)
dataset = TensorDataset(x_torch,y_torch)
seed = 42
torch.manual_seed(seed)
class LinearModel(nn.Module):
    total_loss = 0
    def __init__(self):
        super(LinearModel,self).__init__()
        self.Linear = nn.Linear(1,1)
        self.Linear1 = nn.Linear(2,1)
    def forward(self,x):
        x = self.Linear(x)
        # x = self.Linear1(x)
        return  x
model = LinearModel()
print(model)
print(summary(model,(1,)))

