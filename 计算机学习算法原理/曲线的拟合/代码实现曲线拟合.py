import numpy as  np
import matplotlib.pyplot as  plt
#输入数据集
data = np.array([[0.8, 0], [1.1, 0], [1.7, 0], [3.2, 1], [3.7, 1], [4.0, 1], [4.2, 1]])
x_data = data[:,0]
y_data = data[:,1]
x_train = np.array(x_data)
y_train = np.array(y_data)
#得出激活函数
def sigmoid(x):
    return  1/(1+np.exp(-x))
#初始化
w = 0
b = 0
lr = 0.5
fig, (ax1, ax2) = plt.subplots(2, 1)
epoch_list = []
loss_list = []
#迭代
epocs = 2000
for epoc in range(1,epocs+1):
    # 后向传播
    z = w * x_train + b
    a = sigmoid(z)
    deda = -2 * (y_train - a)
    dadz = a * (1 - a)
    # 计算w的梯度
    dzdw = x_train
    gw = np.mean(deda * dadz * dzdw)
    dzdb = 1
    gb = np.mean(deda * dadz * dzdb)
    w = w - lr * gw
    b = b - lr * gb
    loss = np.mean((y_train-a)**2)
    epoch_list.append(epoc)
    loss_list.append(loss)
    # 显示
    if epoc %50 == 0 or epoc ==1:
        print(f"epoch:{epoc}, loss:{loss}")
        # 9、显示图像
        x_min = x_data.min()
        x_max = x_data.max()
        x_values = np.linspace(x_min, x_max, int((x_max - x_min) * 10))
        y_values = np.round(sigmoid(w * x_values + b), 3)
        ax1.clear()
        ax1.scatter(x_data, y_data)
        ax1.plot(x_values, y_values, c="r")

        ax2.clear()
        ax2.plot(epoch_list, loss_list, c="g")
        plt.pause(1)
plt.show()


