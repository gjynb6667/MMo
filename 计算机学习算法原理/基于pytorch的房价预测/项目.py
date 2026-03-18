import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# -------------------------- 1. 读取Excel数据（核心：pandas.read_excel） --------------------------
file_path = "./Real estate valuation data set.xlsx"
df = pd.read_excel(file_path)

# 数据预处理
df.columns = ["No", "交易日期", "房屋年龄", "到捷运距离", "便利店数", "纬度", "经度", "房价"]
df = df.dropna().drop("No", axis=1)  # 删除空值和无关列

# 特征和标签分离
X = df.drop("房价", axis=1).values
y = df["房价"].values

# -------------------------- 2. 数据划分与标准化 --------------------------
# 8:2划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（线性回归必备）
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# -------------------------- 3. 构建PyTorch线性回归模型 --------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # 6个特征→1个房价

    def forward(self, x):
        return self.linear(x).flatten()


# 初始化模型、损失函数（仅用MSE）、优化器
input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)
criterion = nn.MSELoss()  # 仅使用MSE作为损失/评估指标
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# -------------------------- 4. 训练模型 --------------------------
epochs = 300
train_mse_list = []
test_mse_list = []

for epoch in range(epochs):
    # 训练模式
    model.train()
    optimizer.zero_grad()
    y_train_pred = model(X_train_tensor)
    train_mse = criterion(y_train_pred, y_train_tensor)
    train_mse.backward()
    optimizer.step()

    # 测试模式（eval()评估）
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        test_mse = criterion(y_test_pred, y_test_tensor)

    # 记录MSE
    train_mse_list.append(train_mse.item())
    test_mse_list.append(test_mse.item())

    # 每50轮打印MSE
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], 训练MSE: {train_mse.item():.4f}, 测试MSE: {test_mse.item():.4f}")

# -------------------------- 5. 最终评估（仅用MSE） --------------------------
model.eval()
with torch.no_grad():
    # 反标准化，还原为原始房价单位
    y_train_pred_scaled = model(X_train_tensor).numpy()
    y_test_pred_scaled = model(X_test_tensor).numpy()
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# 计算原始尺度的MSE（核心评估指标）
train_mse_final = mean_squared_error(y_train, y_train_pred)
test_mse_final = mean_squared_error(y_test, y_test_pred)

# 打印最终MSE结果
print("\n========== 模型最终评估（仅MSE） ==========")
print(f"训练集MSE: {train_mse_final:.2f}")
print(f"测试集MSE: {test_mse_final:.2f}")

# -------------------------- 6. 可视化（散点图 + 真实值vs预测值） --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False

# 子图1：MSE损失曲线；子图2：真实值vs预测值；子图3：核心特征散点图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 子图1：训练/测试MSE曲线
axes[0].plot(range(epochs), train_mse_list, label="训练MSE", color="blue")
axes[0].plot(range(epochs), test_mse_list, label="测试MSE", color="red")
axes[0].set_title("训练/测试MSE曲线", fontsize=12)
axes[0].set_xlabel("迭代轮数")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)

# 子图2：测试集真实值vs预测值（核心对比图）
axes[1].scatter(y_test, y_test_pred, alpha=0.6, color="green")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)  # 完美拟合线
axes[1].set_title(f"测试集：真实值 vs 预测值（MSE={test_mse_final:.2f}）", fontsize=12)
axes[1].set_xlabel("真实房价")
axes[1].set_ylabel("预测房价")
axes[1].grid(True)


plt.tight_layout()
plt.show()
