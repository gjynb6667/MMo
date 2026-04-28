import torch
import numpy as np
from torch import nn

# 1.字符输入
text = "In Beijing Sarah bought a basket of apples In Guangzhou Sarah bought a basket of bananas"

torch.manual_seed(1)

# 3.数据集划分
input_seq = [text[:-1]]
output_seq = [text[1:]]
print("input_seq:", input_seq)
# print("output_seq:", output_seq)

# 4.数据编码：one-hot
chars = set(text)
chars = sorted(chars)
# print("chars:", chars)
# {" ":0, "a":1 }
char2int = {char: ind for ind, char in enumerate(chars)}
# print("char2int:", char2int)
# {0:" ", 1: "a"}
int2char = dict(enumerate(chars))

# 将字符转成数字编码
input_seq = [[char2int[char] for char in seq] for seq in input_seq]
# print("input_seq:", input_seq)
output_seq = [[char2int[char] for char in seq] for seq in output_seq]


# one-hot 编码，pytorch的RNN的输入张量的填充
def one_hot_encode(seq, bs, seq_len, size):
    features = np.zeros((bs, seq_len, size), dtype=np.float32)
    for i in range(bs):
        for u in range(seq_len):
            features[i, u, seq[i][u]] = 1.0
    return torch.tensor(features, dtype=torch.float32)


input_seq = one_hot_encode(input_seq, 1, len(text) - 1, len(chars))
output_seq = torch.tensor(output_seq, dtype=torch.long).view(-1)
print("output_seq:", output_seq)


# 5.定义前向模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, out_size)

    def forward(self, x):
        out, hidden = self.bilstm1(x)
        x = out.contiguous().view(-1, self.hidden_size * 2)
        x = self.fc1(x)
        return x, hidden


model = Model(len(chars), 32, len(chars))

# 6.定义损失函数和优化器
cri = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 7.开始迭代
epochs = 1000
for epoch in range(1, epochs + 1):
    output, hidden = model(input_seq)
    loss = cri(output, output_seq)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 8.显示频率设置
    if epoch == 0 or epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss {loss:.4f}")

# print("input_seq.shape:", input_seq.shape)
# print("hidden.shape:", hidden.shape)
# print("output.shape:", output.shape)
# print("input_w:", model.rnn1.weight_ih_l0.shape)

# 预测下面几个字符
input_text = "In Beijing Sarah bought a basket of"  # re
to_be_pre_len = 20

for i in range(to_be_pre_len):
    chars = [char for char in input_text]
    # print(chars)
    character = np.array([[char2int[c] for c in chars]])
    character = one_hot_encode(character, 1, character.shape[1], 23)
    character = torch.tensor(character, dtype=torch.float32)

    out, hidden = model(character)
    char_index = torch.argmax(out[-1]).item()
    input_text += int2char[char_index]
print("预测到的:", input_text)