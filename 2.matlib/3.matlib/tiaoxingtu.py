import  matplotlib.pyplot as plt
import numpy as  np
labels = ['A','B','C','D','E']#条形坐标
values = [23,45,56,67,78]#条形高度
plt.bar(labels,values,
        width = 0.3,#条形的宽度
        color = 'b',#条形的填充颜色
        edgecolor ='r',#条形边缘的颜色
        linewidth = 2,#条形的线宽
        linestyle = '-',#条形的形状
        alpha = 0.7,#透明度
        hatch = 'x',#条形的填充图案
        align = 'center',#条形与x未知的对齐方式
        label = 'test'#标题
        )
plt.legend()#显示图例
plt.show()