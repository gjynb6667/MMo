# 导入必要的库
import numpy as np
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt


# 1.散点输入
class1_points = np.array([[-2.8, 0.1], [0.4, 2.8], [-0.1, 1.9], [-2.0, 1.5], [-0.4, 0.4], [2.1, -0.0],
                          [-0.8, 0.6], [0.4, -2.5], [0.1, -2.0], [-0.6, 1.6], [-1.5, 2.1], [-0.9, 1.2],
                          [1.3, 1.7], [0.2, -0.0], [0.2, -0.1], [-2.1, -0.8], [-0.7, 1.6], [1.4, -1.2],
                          [-0.8, -1.6], [-1.3, 1.1], [-1.2, -0.1], [-2.9, -0.4], [2.4, -1.2], [-2.7, -1.3],
                          [-1.1, -1.9], [1.4, 0.5], [-1.0, 1.8], [2.2, 0.8], [0.9, 1.9], [1.8, -0.9],
                          [1.4, -0.6], [-0.2, -0.7], [-0.3, -2.4], [-1.5, 0.4], [1.2, -0.5], [-1.8, 1.2],
                          [-0.8, -1.7], [-1.7, -2.3], [-0.6, -0.4], [0.3, 1.8], [-0.9, -1.9], [1.6, -1.6],
                          [0.8, -2.6], [-2.6, 1.2], [1.8, -1.0], [0.2, -0.9], [-0.4, -2.5], [1.5, 1.5],
                          [2.2, -1.3], [-1.4, 1.2], [-0.4, 1.3], [-1.3, -1.2], [-2.2, 0.4], [-0.1, 2.9],
                          [1.5, -2.4], [1.1, 2.3], [0.4, 2.8], [-0.8, -1.2], [-2.7, 0.4], [2.3, 1.1],
                          [0.9, 1.0], [0.9, 0.7], [-1.8, 0.3], [-1.7, -0.4], [1.0, -0.3], [-1.1, -0.6],
                          [-2.4, -0.4], [2.6, -1.4], [1.3, -0.7], [-0.0, 1.0], [-1.1, -2.4], [2.0, -0.5],
                          [0.3, 2.3], [-0.6, -2.8], [0.6, -1.8], [0.9, -0.4], [1.0, -1.3], [-0.4, 0.2],
                          [2.3, 0.1], [2.2, 0.5], [2.5, 0.2], [-2.1, -1.3], [-1.1, 1.0], [1.7, 1.5],
                          [0.9, -1.0], [1.1, -2.2], [-0.2, -2.4], [0.7, -1.1], [-0.4, 0.3], [-0.0, -2.6],
                          [-0.3, -0.1], [-1.8, -1.6], [-0.8, 2.5], [-1.9, -1.4], [-2.5, 1.2], [-2.3, -0.6],
                          [-1.6, 0.1], [-1.9, 0.0], [1.1, -0.6], [-0.2, 2.7]])
class2_points = np.array([[5.1, 1.6], [-5.1, 1.2], [-4.6, 3.0], [-5.7, 0.5], [5.4, 2.5], [-4.5, -2.5],
                          [4.9, -0.4], [1.4, 5.5], [1.4, 5.7], [4.2, -1.0], [-1.3, -4.8], [-4.4, -2.9],
                          [-3.6, 3.4], [-3.4, -4.1], [-5.8, -0.1], [4.7, 3.0], [-1.4, 4.4], [2.5, -4.7],
                          [2.7, 5.3], [4.1, -2.8], [-4.0, -2.5], [5.0, 0.5], [-4.0, 4.3], [5.0, 1.3],
                          [3.3, 3.4], [2.2, 4.6], [2.8, 4.8], [4.0, -3.5], [4.6, -3.1], [0.5, 5.5],
                          [-4.7, 3.1], [-5.7, 1.0], [2.8, -5.1], [-1.3, 4.9], [2.7, 5.2], [-4.9, 1.3],
                          [4.1, -2.9], [-4.9, -3.3], [-4.6, 2.8], [-4.6, 3.1], [-1.8, 4.8], [-2.4, 5.3],
                          [-5.2, 3.0], [-3.7, -4.4], [1.5, -5.0], [4.8, 1.1], [-0.6, -5.8], [0.7, -4.9],
                          [0.2, 5.7], [5.8, -0.5], [-2.0, -4.0], [3.9, -3.1], [0.2, 5.1], [4.5, 1.5],
                          [-1.4, -5.3], [5.0, -1.4], [5.1, 0.7], [5.0, -3.0], [-0.7, -5.1], [5.2, -1.5],
                          [-0.7, 4.5], [2.1, 3.9], [-2.4, 5.0], [-0.8, 4.9], [-5.1, 0.3], [3.3, 3.6],
                          [-0.4, -5.1], [3.8, 4.6], [-5.3, -2.5], [-5.5, -1.2], [0.6, 5.5], [-5.8, 0.8],
                          [-5.3, 1.2], [2.0, -5.5], [5.7, 0.7], [-1.1, -4.7], [-0.0, 5.7], [-3.0, -4.8],
                          [-3.5, -4.0], [4.9, 1.8], [-1.1, -5.5], [-2.7, 4.2], [-4.9, -1.6], [-0.2, -5.2],
                          [2.5, 5.1], [-0.0, 5.4], [3.9, 4.4], [3.5, 4.8], [4.8, -1.9], [5.4, -1.0],
                          [3.7, 4.6], [1.8, 5.2], [4.7, 3.4], [4.1, 2.3], [0.6, -5.4], [1.4, 5.5],
                          [-5.5, 1.3], [4.2, 2.5], [-2.2, 3.9], [5.9, -0.4]])

# 定义点的标签
labels1 = np.ones(len(class1_points))
labels2 = np.zeros(len(class2_points))

# 合并两类点和标签
points = np.concatenate((class1_points, class2_points))
labels = np.concatenate((labels1, labels2))

# 将标签转换为one-hot编码,方便后续训练使用
labels = tf.keras.utils.to_categorical(labels, num_classes=2)

# 2.定义前向模型
input_shape = 2
hidden_layer_shape = 3
output_shape = 2

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_shape, activation='sigmoid', input_shape=(input_shape,)),
    tf.keras.layers.Dense(output_shape, activation='softmax')
])

# 打印模型信息
model.summary()

# 3.定义损失函数和优化器
learning_rate = 0.05
# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam   (learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# 创建等高线绘图的网格点
x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
step_size = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                     np.arange(y_min, y_max, step_size))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 创建三维图形和右侧的二维子图
fig = plt.figure(figsize=(10, 5))
ax_3d = fig.add_subplot(121, projection='3d')
ax_2d = fig.add_subplot(122)

# 设置俯视视角
ax_3d.view_init(elev=30, azim=-60)  # 俯视视角，elev为俯仰角，azim为方位角

# 4.开始迭代
num_iterations = 1000
# 5.显示频率设置
frequency_display = 20
for n in range(0, num_iterations + 1, frequency_display):
    model.fit(points, labels, epochs=frequency_display, verbose=0)
    # 6.显示与输出
    # 使用训练好的模型预测网格点的标签
    Z = model.predict(grid_points)
    Z = Z[:, 1]  # 取正类的概率值

    ax_3d.cla()
    ax_2d.cla()
    # 绘制点
    ax_3d.scatter(class1_points[:, 0], class1_points[:, 1], np.ones_like(class1_points[:, 0]), c='blue',
                  label='Class 1')
    ax_3d.scatter(class2_points[:, 0], class2_points[:, 1], np.zeros_like(class2_points[:, 0]), c='red',
                  label='Class 2')

    # 绘制分类的超平面
    ax_3d.plot_surface(xx, yy, np.reshape(Z, xx.shape), alpha=0.5)

    # 在高度值为0.5处添加决策边界
    ax_3d.contour(xx, yy, np.reshape(Z, xx.shape), levels=[0.5], colors='black')

    ax_3d.set_xlabel('feature 1')
    ax_3d.set_ylabel('feature 2')
    ax_3d.set_zlabel('label')
    ax_3d.set_title('hyperplane')

    # 绘制散点（二维图）
    ax_2d.scatter(class1_points[:, 0], class1_points[:, 1], c='blue', label='Class 1')
    ax_2d.scatter(class2_points[:, 0], class2_points[:, 1], c='red', label='Class 2')
    # 绘制等高线图，并将决策边界设为0.5的高度值
    plt.contour(xx, yy, np.reshape(Z, xx.shape), levels=[0.5], colors='black')  # 在高度值为0.5处添加决策边界
    plt.title(f"Epochs: {n}")
    plt.pause(0.2)