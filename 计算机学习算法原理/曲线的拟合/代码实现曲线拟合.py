import numpy as  np
#输入数据集
data = np.array([[0.8, 0], [1.1, 0], [1.7, 0], [3.2, 1], [3.7, 1], [4.0, 1], [4.2, 1]])
x_data = data[:,0]
y_data = data[:,1]
x_train = np.array(x_data)
y_train = np.array(y_data)
#得出激活函数
def sigmoid(x):
    return  1/1+np.exp(x)
#初始化
w = 0
b = 0
lr = 0.01
#迭代
epocs = 2000
for epoc in epocs:
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
    w_new = w - lr * gw
    b_new = b - lr * gb
    loss = np.mean((x_train-a)**2)
    # 显示
    if epoc %50 == 0 or epoc ==1:
        



