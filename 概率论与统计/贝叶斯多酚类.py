import numpy as np
import matplotlib.pyplot as plt


def pdf(x, mean, cov):
    # 获取均值向量的长度，即特征的数量
    n = len(mean)
    # 计算PDF的系数部分
    coeff = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(np.linalg.det(cov)))
    # 计算PDF的指数部分
    exponet = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponet)


# 1.散点输入
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

class3_points = np.array([[3.3, 1.2],
                          [3.8, 0.9],
                          [3.3, 0.6],
                          [2.8, 1.3],
                          [3.5, 0.6],
                          [4.2, 0.3],
                          [3.1, 0.9]])

# 合并数据集、创造标签
X = np.concatenate((class1_points, class2_points, class3_points))
y = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points)), np.ones(len(class3_points)) + 1))

# 2. 计算先验概率(每一个类别的数据在数据集中的比例)
prior_probabilities = [np.sum(y == 0) / len(y), np.sum(y == 1) / len(y), np.sum(y == 2) / len(y)]
print(prior_probabilities)

# 3.计算高斯分布的概率密度函数
# 求解每个类别的均值
class_means = [np.mean(X[y == 0], axis=0), np.mean(X[y == 1], axis=0), np.mean(X[y == 2], axis=0)]
# print(class_means)

# 求解每个类别的协方差矩阵
X_y_0 = X[y == 0].T
X_y_1 = X[y == 1].T
X_y_2 = X[y == 2].T
class_covs = [np.cov(X_y_0), np.cov(X_y_1), np.cov(X_y_2)]

xx, yy = np.meshgrid(np.arange(0, 5, 0.05), np.arange(0, 4, 0.05))
# 预测网格点
grid_points = np.c_[xx.ravel(), yy.ravel()]

grid_label = []
# 每一组网格点进行预测
for point in grid_points:
    posterior_probabilities = []
    for i in range(3):
        # 使用高斯分布的概率密度函数求解条件概率
        likelihood = pdf(point, class_means[i], class_covs[i])
        # 4.得到后验概率，比较大小，获得分类
        posterior_probabilities.append(prior_probabilities[i] * likelihood)
    # 比较大小，获得分类
    pre_class = np.argmax(posterior_probabilities)
    grid_label.append(pre_class)

# 5. 显示决策边界
# 预测的标签与xx他们形状一致
grid_label = np.array(grid_label).reshape(xx.shape)

plt.scatter(class1_points[:, 0], class1_points[:, 1], c="blue", label="class 1")
plt.scatter(class2_points[:, 0], class2_points[:, 1], c="red", label="class 2")
plt.scatter(class3_points[:, 0], class3_points[:, 1], c="green", label="class 3")

contour = plt.contour(xx, yy, grid_label, colors="black")

plt.legend()
plt.show()