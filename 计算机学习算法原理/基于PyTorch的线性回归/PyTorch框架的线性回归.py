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
# def function_loss(X,Y,w,b):
#     y = np.dot(X,w)+b
#     loss = np.mean((2*(y-Y)**2))
#     return loss
#
# gd_path = []
# # 构建网格点
# w_values = np.linspace(-20, 80, 100)
# b_values = np.linspace(-20, 80, 100)
# W, B = np.meshgrid(w_values, b_values)
# loss_values = np.zeros_like(W)
#
# for i, w in enumerate(w_values):
#     for j, b in enumerate(b_values):
#         loss_values[j, i] = function_loss(x_data, y_data, w, b)
#
#
# # 创建图形对象和子图布局
# fig = plt.figure(figsize=(12, 6))
# gs = gridspec.GridSpec(2, 2)
#
# # 左上格子
# ax2 = fig.add_subplot(gs[0, 0])
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")
# ax2.set_title("Data")
#
# # 左下格子
# ax3 = fig.add_subplot(gs[1, 0])
# ax3.set_xlabel("w")
# ax3.set_ylabel("b")
# ax3.set_title("Contour Plot")
#
# # 整个右侧格子
# ax1 = fig.add_subplot(gs[:, 1], projection='3d')
# ax1.plot_surface(W, B, loss_values, cmap='viridis', alpha=0.8)
# ax1.set_xlabel('w')
# ax1.set_ylabel('b')
# ax1.set_zlabel('Loss')
# ax1.set_title("Surface Plot")
# #定义模型
# model  =  nn.Linear(1,1)
# w = float(model.weight)
# b = float(model.bias)
# w_temp = w
# b_temp = b
# w = w_temp
# b = b_temp
#其他方案
#Sequential是一个模块容器 内涵forword方法能够自动排序
# model = nn.Sequential(nn.Linear(1,1))
#模拟forword的方法自写一个类
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel,self).__init__()
#         self.layers = nn.ModuleList([nn.Linear(1,1)])
#     def forword(self,x):
#         for layer in self.layers:
#             x = layer
#         return x
#model = nn.ModuleDict({"linear"})
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel,self).__init__()
#         self.layers = nn.ModuleDict({"linear":nn.Linear(1,1)})
#     def forword(self,x):
#         for layer in self.layers.values():
#             x = layer
#         return x
#最常用的
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
# 获取当前的参数值
# w = float(model.weight)
# b = float(model.bias)
# 5、显示频率设置
#
# 根据当前参数拟合直线
# x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
# y_line = np.dot(x_line, w) + b

#         # 更新子图 1 数据并绘制
#         ax2.clear()
#         ax2.scatter(x_data, y_data)
#         ax2.plot(x_line, y_line, '-')
    #         ax2.set_title(f"Linear Regression: w={w}, b={b}")
    #         # 绘制当前w和b的位置
    #         ax1.scatter(w, b, function_loss(x_data, y_data, w, b), c='black', s=20)
    #
    #         # 绘制俯视图等高线
    #         ax3.clear()
    #         ax3.contourf(W, B, loss_values, levels=20, cmap='viridis')
    #         ax3.scatter(w, b, c='black', s=20)
    #
    #         # 绘制梯度下降路径
    #         if len(gd_path) > 0:
    #             gd_w, gd_b = zip(*gd_path)
    #             ax1.plot(gd_w, gd_b,
    #                      [function_loss(x_data, y_data, np.array(gd_w[i]), np.array(gd_b[i])) for i in
    #                       range(len(gd_w))],
    #                      c='black')
    #             ax3.plot(gd_w, gd_b)
    #         plt.pause(1)
    # plt.show()


