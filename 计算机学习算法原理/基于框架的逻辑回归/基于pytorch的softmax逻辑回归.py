import numpy as np
import torch
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

X_train = np.concatenate((class1_points, class2_points))
Y_train = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points))))

# 2.定义前向模型
torch.manual_seed(1)


class LogisticRegreModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegreModel, self).__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc(x)
        # dim 指定了softmax在哪个维度上进行
        # dim = 1第二个维度上进行，列
        # example：x  (sample， num_classes)
        # torch.softmax(x, dim=1), pytorch在每个样本上的类别进行softmax，保证每个样本的所有类别的概率和为1。
        return torch.softmax(x, dim=1)


model = LogisticRegreModel()

# 3.定义损失函数 和优化器
cri = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

plt.figure(figsize=(12, 6))

xx, yy = np.meshgrid(np.arange(0, 6, 0.1), np.arange(0, 6, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 4.开始迭代
epoches = 1000
for epoch in range(1, epoches + 1):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(Y_train, dtype=torch.long)

    outputs = model(inputs)
    loss = cri(outputs, labels)
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5.显示频率设置
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch: {epoch}, loss: {loss}")
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        Z = model(grid_tensor).detach().numpy()
        Z = Z[:, 1]
        Z = Z.reshape(xx.shape)

        plt.cla()
        plt.scatter(class1_points[:, 0], class1_points[:, 1])
        plt.scatter(class2_points[:, 0], class2_points[:, 1])
        plt.contour(xx, yy, Z, levels=[0.5])
        plt.pause(1)

plt.show()