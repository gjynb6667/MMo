# 导入必要的库
import tensorflow as tf
import numpy as np
from IPython import display
import matplotlib.pyplot as plt

# 1.散点输入
points = np.array([[-0.5, 7.7], [1.2, 65.8], [0.4, 39.2], [-1.4, -15.7], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]])

# 分离训练集的特征与标签
x_train = points[:, 0]
y_train = points[:, 1]

tf.random.set_seed(4)

# 2.定义前向模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,), kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dense(1)
])

# 4.定义损失函数和优化器
loss_function = 'mean_squared_error'
learning_rate = float("{}".format(0.005))
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 编译模型
model.compile(optimizer=optimizer, loss=loss_function)

# 创建子图对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 设置坐标范围
x_min, x_max = -2, 2
x_values = np.linspace(x_min, x_max, 40)

# 迭代次数列表，方便后续绘图
step_list = []
# 训练损失列表，方便后续绘图
train_loss_list = []

# 5.开始迭代和显示
num_iterations = 1000
frequency_display = 20

x_train_tensor = tf.constant(x_train, dtype=tf.float32)
y_train_tensor = tf.constant(y_train, dtype=tf.float32)

for n in range(0, num_iterations + 1, frequency_display):
    # 开始训练
    history = model.fit(x_train_tensor, y_train_tensor, epochs=frequency_display, verbose=0)
    # 获取训练损失
    train_loss = history.history['loss'][0]

    # 进行预测
    y_predict = model.predict(x_values)
    y_values = np.round(y_predict, 3)

    # 将损失值与迭代次数添加到列表中，方便绘图
    train_loss_list.append(train_loss)
    step_list.append(n)

    # 绘制第一个子图
    ax1.clear()
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-60, 110)
    ax1.set_xlabel("x axis label")
    ax1.set_ylabel("y axis label")
    ax1.scatter(x_train, y_train)
    ax1.plot(x_values, y_values, 'r')  # 绘制拟合线

    ax2.clear()
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.plot(step_list, train_loss_list, 'r-', label="Train Loss")
    ax2.legend()
    plt.pause(0.2)