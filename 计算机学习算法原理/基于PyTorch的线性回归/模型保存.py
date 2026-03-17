import  torch
import numpy as np
from matplotlib import gridspec
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader

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
        self.Linear2 = nn.Linear(1,1)
    def forword(self,x):
        x = self.Linear(x)
        return  x
model = LinearModel
criteion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
epoches = 500
dataloader = DataLoader(dataset,batch_size=5,shuffle=True)
for n in range(1,epoches+1):
    # gd_path.append((w, b))
    total_loss = 0
    for x,y in dataloader:
        y_hat = model(x.unsqueeze(1))
        print(y_hat.shape)
        loss = criteion(y_hat.squeeze(1), y)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss/len(dataloader)
    if n % 10 == 0 or n == 1:
        print(f"epoches:{n},avg_loss:{avg_loss}")
#保存参数
torch.save(model.state_dict(),'model.pth')
model.load_state_dict()
#保存模型
torch.save(model,'entire_model.pth')
torch.load()
#模型评估
model.eval()
x_test = torch.tensor([[1.2],[3.4]],dtype=torch.float32)
with torch.no_grad():
    y_pred = model(x_test)