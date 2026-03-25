import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

X_train = np.concatenate((class1_points, class2_points))
Y_train = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points))))

# 2.定义前向模型
model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid", input_shape=(2,))])

# 3.定义损失函数 和优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
epoch_list = []
loss_list = []
# 4.开始迭代
epoches = 1000
for epoch in range(1, epoches + 1, 50):
    history = model.fit(X_train, Y_train, epochs=50, verbose=0)
    # 获取损失
    loss = history.history['loss'][0]

    # 获取w1w2b
    w1, w2 = model.get_weights()[0].flatten()
    b = model.get_weights()[1][0]

    # 5.显示频率设置
    print(f"epoch: {epoch}, loss: {loss}")

    # 斜率和截距求解
    slope = -w1 / w2
    intercept = -b / w2
    # 绘制直线
    x_min, x_max = 0, 5
    x = np.array([x_min, x_max])
    y = slope * x + intercept

    # 6.绘制
    ax1.clear()
    ax1.scatter(class1_points[:, 0], class1_points[:, 1])
    ax1.scatter(class2_points[:, 0], class2_points[:, 1])
    ax1.plot(x, y)

    ax2.clear()
    epoch_list.append(epoch)
    loss_list.append(loss)
    ax2.plot(epoch_list, loss_list)
    plt.pause(1)
plt.show()