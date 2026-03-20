import numpy as np
import  matplotlib.pyplot as plt
def pdf(x, mean, cov):
    # 获取均值向量的长度，即特征的数量
    n = len(mean)
    # 计算PDF的系数部分
    coeff = 1 / ((2 * np.pi) ** (n/2) * np.sqrt(np.linalg.det(cov)))
    # 计算PDF的指数部分
    exponet = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponet)

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
#创建数据集和标签
X = np.concatenate((class1_points,class2_points))
Y = np.concatenate((np.zeros(len(class1_points)),np.ones(len(class2_points))))
#计算先验概率
p = [np.sum(Y==0)/len(Y),np.sum(Y==1)/len(Y)]
print(p)
#求均值
y_mean = [np.mean(X[Y ==0],axis=  0),np.mean(X[Y==1],axis=0)]
#进行转置
x = X[Y==0].T
y = X[Y==1].T
class_covs = [np.cov(x),np.cov(y)]
point = np.array([0.5,3])
xx,yy = np.meshgrid(np.arange(0,5,0.05),np.arange(0,4,0.05))
grid_point = np.c_[xx.ravel(),yy.ravel()]
grid_label = []
for point in grid_point:
    posterior_provavilities = []
    for i in range(2):
        pdf(point, y_mean[i], class_covs[i])
        likelihood = pdf(point, y_mean[i], class_covs[i])
        print(likelihood)
        posterior_provavilities.append(p[i] * likelihood)
    pre_class = np.argmax(posterior_provavilities)
    grid_label.append(pre_class)

# 5. 显示决策边界
# 预测的标签与xx他们形状一致
grid_label = np.array(grid_label).reshape(xx.shape)
#画出散点图
plt.scatter(class1_points[:,0],class1_points[:,1],color = 'r',label = 'class 1')
plt.scatter(class2_points[:,0],class2_points[:,1],color = 'b',label = 'class 2')
contour = plt.contour(xx, yy, grid_label, levels=[0.5], colors="green")
if pre_class == 0:
    plt.text(point[0]+0.1,point[1]-0.1,'class 1')
elif pre_class == 1:
    plt.text(point[0]+0.1,point[1]-0.1,'class 2')
plt.legend()
plt.show()