import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
df_train = pd.read_csv("./guoiao/train/moutai_train.csv")
df_test = pd.read_csv("./guoiao/test/moutai_test.csv")

# 数据预处理
df_train = df_train.dropna().drop("Date", axis=1)
df_test = df_test.dropna().drop("Date", axis=1)


# 分离输入(OHLC)和输出(Adj Close)
input_features = df_train[['High', 'Low', 'Close']].values
output_target = df_train[['Adj Close']].values
test_input_features = df_test[['High', 'Low', 'Close']].values
test_output_target = df_test[['Adj Close']].values

print(f"输入范围: {input_features.min():.2f} - {input_features.max():.2f}")
print(f"输出范围: {output_target.min():.2f} - {output_target.max():.2f}")

# ========== 关键：分别归一化 ==========
scaler_X = StandardScaler()
scaler_y = StandardScaler()

input_scaled = scaler_X.fit_transform(input_features)
test_input_scaled = scaler_X.transform(test_input_features)

output_scaled = scaler_y.fit_transform(output_target)
test_output_scaled = scaler_y.transform(test_output_target)


# ========== 创建时间序列滑动窗口 ==========
def create_sequences(X, y, seq_length):
    """
    使用过去seq_length天的OHLC数据，预测当天的Adj Close
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        # 过去seq_length天的OHLC
        X_seq.append(X[i:i + seq_length])
        # 预测第i+seq_length天的Adj Close
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


# 尝试不同的序列长度
seq_length = 10  # 使用过去20天的数据预测下一天
X_seq, y_seq = create_sequences(test_input_scaled, test_output_scaled, seq_length)

print(f"\n序列数据形状: {X_seq.shape}")  # (样本数, 20, 4)
print(f"标签形状: {y_seq.shape}")  # (样本数, 1)

# 转换为tensor
input_tensor = torch.tensor(X_seq, dtype=torch.float32)
output_tensor = torch.tensor(y_seq, dtype=torch.float32)


# ========== 改进的LSTM模型 ==========
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

#
# 模型参数
input_size = 3
hidden_size = 128  # 增加隐藏层大小
num_layers = 2
output_size = 1
#
model = StockLSTM(input_size, hidden_size, num_layers, output_size, dropout=0.3)

# ========== 划分训练集和验证集 ==========
train_size = int(0.8 * len(input_tensor))
val_size = len(input_tensor) - train_size

train_X, val_X = input_tensor[:train_size], input_tensor[train_size:]
train_y, val_y = output_tensor[:train_size], output_tensor[train_size:]

print(f"训练集大小: {len(train_X)}, 验证集大小: {len(val_X)}")

# ========== 训练配置 ==========
model.load_state_dict(torch.load("best_stock_model.pth"))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

# ========== 训练循环 ==========
epochs = 300
batch_size = 64
train_losses = []
val_losses = []
best_val_loss = float('inf')

# print("\n开始训练...")
# for epoch in range(1, epochs + 1):
#     # 训练阶段
#     model.train()
#     total_train_loss = 0
#     num_batches = 0
#
#     # 小批量训练
#     for i in range(0, len(train_X), batch_size):
#         batch_X = train_X[i:i + batch_size]
#         batch_y = train_y[i:i + batch_size]
#
#         optimizer.zero_grad()
#         output = model(batch_X)
#         loss = criterion(output, batch_y)
#         loss.backward()
#
#         # 梯度裁剪
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
#         optimizer.step()
#         total_train_loss += loss.item()
#         num_batches += 1
#
#     avg_train_loss = total_train_loss / num_batches
#     train_losses.append(avg_train_loss)
#
#     # 验证阶段
#     model.eval()
#     with torch.no_grad():
#         val_output = model(val_X)
#         val_loss = criterion(val_output, val_y)
#         val_losses.append(val_loss.item())
#
#     # 学习率调度
#     scheduler.step(val_loss)
#
#     # 保存最佳模型
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'best_stock_model.pth')
#
#     # 打印进度
#     if epoch % 20 == 0:
#         current_lr = optimizer.param_groups[0]['lr']
#         print(
#             f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss.item():.6f} | LR: {current_lr:.6f}")
#
# print(f"\n最佳验证损失: {best_val_loss:.6f}")

# ========== 预测和评估 ==========
model.eval()
with torch.no_grad():
    test_pred_scaled = model(input_tensor)
    print(test_pred_scaled)


    # 反归一化到原始价格
    test_pred = scaler_y.inverse_transform(test_pred_scaled.numpy())
    test_true = scaler_y.inverse_transform(output_tensor.numpy())


    # 计算RMSE和MAPE
    test_rmse = np.sqrt(np.mean((test_pred - test_true) ** 2))

    test_mape = np.mean(np.abs((test_true - test_pred) / test_true)) * 100

    print(f"\n训练集 RMSE: ${test_rmse:.2f}, MAPE: {test_mape:.2f}%")

# ========== 可视化结果 ==========
fig, axes = plt.subplots(1, 2, figsize=(15, 10))



# 2. 预测 vs 真实值（验证集）
axes[0].plot(test_true[-200:], label='True Values', alpha=0.7)
axes[0].plot(test_pred[-200:], label='Predictions', alpha=0.7)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Adjusted Close Price ($)')
axes[0].set_title('Predictions vs True Values (Last 200 days)')
axes[0].legend()
axes[0].grid(True)

# 3. 散点图
axes[1].scatter(test_true, test_pred, alpha=0.5, s=10)
axes[1].plot([test_true.min(), test_true.max()], [test_true.min(), test_true.max()], 'r--', lw=2)
axes[1].set_xlabel('True Values ($)')
axes[1].set_ylabel('Predictions ($)')
axes[1].set_title(f'Prediction Scatter Plot (RMSE: ${test_rmse:.2f})')
axes[1].grid(True)
plt.show()


