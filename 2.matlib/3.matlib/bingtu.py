import  matplotlib.pyplot as plt
import  numpy as np
sizes = [25,35,20,20]
labels = ['A','B','C','D']
colors = ['gold','yellowgreen','lightcoral','lightskyblue']
plt.pie(sizes,
        explode=[0,0,0,0],#用于指定每个扇形是否突出
        labels = labels,#用于制定每个扇形的标签
        colors = colors,#用于指定每一个闪现的颜色
        autopct = '%1.1f%%',#用于在每个扇形显示它所占的百分比
        startangle = 140,#饼图开始的角度默认从0开始
        shadow = False,#用于指定是否为饼图添加阴影
        radius = 1,#设置饼图的半径
        wedgeprops = dict(edgecolor = 'black',linewidth = 2,linestyle = '-'),#用于设定饼图中每个扇形的属性
        textprops= dict(color = 'red',weight = 'bold'),#用于设定饼图中标签的文本属性
        center = (0,0), #用于指定饼图的中心位置
        frame = False, #用于是否指定为饼图添加一个框
        )
plt.show()