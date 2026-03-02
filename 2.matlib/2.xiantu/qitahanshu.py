import  matplotlib.pyplot as plt
import  numpy as np
x = np.arange(0,3*np.pi,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.subplot(2,1,1)#2行1列第一个图
plt.plot(x,y_sin)#创建线图
plt.title('Sine Wave')#创建标题
plt.xlabel('x')#创建横坐标标题
plt.ylabel('y_sin')#创建纵坐标标题
plt.subplot(2,1,2)#2行1列第二个图
plt.plot(x,y_cos)#创建线图
plt.title('Cos Wave')#创建标题
plt.xlabel('x')#创建横坐标标题
plt.ylabel('y_cos')#创建纵坐标标题
plt.tight_layout()#来自动调整子图参数
plt.show()
