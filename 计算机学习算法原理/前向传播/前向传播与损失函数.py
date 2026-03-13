import  numpy as np
import  matplotlib.pyplot as  plt
#数据集
data = [[0.8,1.0],
        [1.7,0.9],
        [2.7,2.4],
        [3.2,2.9],
        [3.7,2.8],
        [4.2,3.8],
        [4.2,2.7]]
data = np.array(data)
#得到x,y坐标
x_data = data[:,0]
y_data = data[:,1]
#前向计算
w = 0
b = 0
ycha = w*x_data + b
#单点误差
e = y_data - ycha
#均方误差
e1 = np.mean((y_data-ycha)**2)
#绘图
fig = plt.figure(figsize= (10,5))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
ax1.set_xlim(0,5)
ax1.set_ylim(0,6)
ax1.set_xlabel("x坐标")
ax1.set_ylabel("y坐标")
ax1.scatter(x_data,y_data,color = "b")
y_lower = w*0+b
y_upper = w*5+b
ax1.plot([0,5],[y_lower,y_upper],color = "r",linewidth = 3)
plt.show()
