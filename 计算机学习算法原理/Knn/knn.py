from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
#1.定义数据集
#创建数据集
point1 = [[7.7, 6.1], [3.1, 5.9], [8.6, 8.8], [9.5, 7.3], [3.9, 7.4], [5.0, 5.3], [1.0, 7.3]]
point2 = [[0.2, 2.2], [4.5, 4.1], [0.5, 1.1], [2.7, 3.0], [4.7, 0.2], [2.9, 3.3], [7.3, 7.9]]
point3 = [[9.2, 0.7], [9.2, 2.1], [7.3, 4.5], [8.9, 2.9], [9.5, 3.7], [7.7, 3.7], [9.4, 2.4]]
#对特征点进行合并concatenate (1.数据集,2.轴)
data = np.concatenate((point1,point2,point3),axis=0)
#构建标签
data_label = np.array([0]*len(point1)+[1]*len(point2)+[2]*len(point3))
#2.创建KNN算法进行训练
#构建knn算法
knn_suan = KNeighborsClassifier(3)
#进行训练
knn_suan.fit(data,data_label)
#3.设定未知点
#1.设定坐标点网络
axis = [0,10,0,10]
#生成坐标点网络
x0,x1 = np.meshgrid(
    np.linspace(axis[0],axis[1],100),
    np.linspace(axis[0],axis[1],100)
)
axis_xy = np.c_[x0.ravel(),x1.ravel()]
#4.knn的预测和绘制决策边界
#等高线的预测
y_predict = knn_suan.predict(axis_xy)
y_predict = y_predict.reshape(x0.shape)
#等高线的绘制
plt.contour(x0,x1,y_predict)
plt.scatter(data[data_label == 0 , 0],data[data_label == 0, 1],marker = "^")
plt.scatter(data[data_label == 1 , 0],data[data_label == 1, 1],marker = "*")
plt.scatter(data[data_label == 2 , 0],data[data_label == 2, 1],marker = "1")
plt.show()




