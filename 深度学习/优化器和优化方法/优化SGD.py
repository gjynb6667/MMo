# 导入相关库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1.散点输入
points = np.array(
    [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6],
     [0.4, 34.0], [0.8, 62.3]])
# 分离特征和标签
X = points[:, 0]
Y = points[:, 1]

# 2.参数初始化
w = 0
b = -1
lr = 0.001
def loss_func(x,w,b):
    pre_y = np.dot(x,w)+b
    loss = np.mean((Y-pre_y)**2)
    return loss
def SGD(points,w,b,batch_size):
    np.random.shuffle(points)
    for num in range(0,len(points),batch_size):
        get_point = points[num:num+batch_size,:]
        get_x = get_point[:,0]
        get_y = get_point[:,1]
        pre_y = w*get_x+b
        dw = np.mean(2*(pre_y-get_y)*get_x)
        db = np.mean(2*(pre_y-get_y)*get_x)
        w -= lr*dw
        b -= lr*db
    return w,b
w_value = np.linspace(-20,80,100)
b_value = np.linspace(-20,80,100)
W,B = np.meshgrid(w_value,b_value)
loss_value = np.zeros_like(W)
for i,w in enumerate(w_value):
    for j,b in enumerate(b_value):
        loss_value[i,j] = loss_func(X,w,b)

epochs = 1000
bs = 2
for epoch in range(1,epochs+1):
    w,b = SGD(points,w,b,bs)
    if epoch ==1 or epoch%20 ==0:
        print(loss_func(X,w,b))



