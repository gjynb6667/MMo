import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,3*np.pi,0.1)#从0开始到3π结束 步长是0.1
y = np.sin(x)#基于x创建Y
colors = y#基于y值来映射颜色
plt.scatter(x,y,
            s = 10,     #散点的大小
            c = colors, #散点的颜色，这里使用y值映射颜色
            marker = 'o',#散点的样式标记
            cmap = 'viridis',#颜色映射
            norm = None, #默认的标准化
            vmin = -1,#颜色映射的最小值
            vmax = 1,#颜色映射的最大值
            alpha = 0.5,#透明度
            linewidths = 0.5,#散点边缘的线宽
            edgecolors = 'w'#三点边缘的颜色
)
plt.colorbar()#显示颜色条
plt.show()