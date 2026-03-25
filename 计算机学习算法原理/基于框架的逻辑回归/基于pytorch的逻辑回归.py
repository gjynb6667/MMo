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
x_data = np.concatenate((class1_points,class2_points))
y_data = np.concatenate((np.zeros(len(class1_points)),np.ones(len(class2_points))))
torch.manual_seed(1)
class linear(nn.Module):
    def __init__(self):
        super(linear,self).__init__()
        self.fc = nn.Linear(2,1)
    def forward(self,x):
        x = self.fc(x)
        return torch.sigmoid(x)
model = linear()
cri = torch.nn.BCELoss()
optimzier = torch.optim.SGD(model.parameters(),lr = 0.05)
epochs = 1000
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
epoch_list = []
loss_list = []
for epoc in range(1,epochs+1):
    input = torch.tensor(x_data,dtype=torch.float32)
    label = torch.tensor(y_data,dtype=torch.float32).unsqueeze(1)
    outputs = model(input)
    loss = cri(outputs,label)
    optimzier.zero_grad()
    loss.backward()
    optimzier.step()

    if epoc%50 == 0 or epoc ==1:
        print(f'epoch:{epoc},loss:{loss}')
        w1,w2 = model.fc.weight.data.flatten()
        b = model.fc.bias.data[0]
        slope = -w1/w2
        intercept = -b/w2
        x_min,x_max = 0,5
        x = np.array([x_min,x_max])
        y = slope*x+intercept
        ax1.clear()
        ax1.scatter(class1_points[:,0],class1_points[:,1])
        ax1.scatter(class2_points[:,0],class2_points[:,1])
        ax1.plot(x,y)
        ax2.clear()
        epoch_list.append(epoc)
        loss_list.append(loss.tolist())
        ax2.plot(epoch_list, loss_list)
        plt.pause(1)
plt.show()