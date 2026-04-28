import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
import re

# 数据预处理
corpus = [
    "jack like dog", "jack like cat", "jack like animal",
    "dog cat animal", "banana apple cat dog like", "dog fish milk like",
    "dog cat animal like", "jack like apple", "apple like", "jack like banana",
    "apple banana jack movie book music like", "cat dog hate", "cat dog like"
]


# 将句子分词，并转换为小写
def tokenize(sentence):
    # \b 单词的边界
    # \w+ 匹配一个或者多个单词字符（字母，数字，下划线）
    # \[,.!?] 匹配逗号、句号、感叹号和问号
    word_list = []
    for word in re.findall(r"\b\w+\b|[,.!?]", sentence):
        word_list.append(word.lower())
    return word_list


words = []
for sentence in corpus:
    for word in tokenize(sentence):
        words.append(word)

# print(words)
word_counts = Counter(words)

vocab = sorted(word_counts, key=word_counts.get, reverse=True)
# print(vocab)

# {"like": 1, "dog": 2}
# 创建词汇表到索引的隐射
vocab2int = {word: ii for ii, word in enumerate(vocab, 1)}
# print(vocab2int)
# 将所有的单词转换为索引表示
int2vocab = {ii: word for ii, word in enumerate(vocab, 1)}
# print(int2vocab)

# 将所有单词变成索引
word2index = [vocab2int[word] for word in words]
print(word2index)
# print(word2index)
window = 1
center = []
context = []

for i, target in enumerate(word2index[window: -window], window):
    # print(i, target)
    # 数据：3 1 2 3 1 4
    # 索引：0 1 2 3 4 5
    center.append(target)
    context.append(word2index[i - window: i] + word2index[i + 1: i + 1 + window])

torch.manual_seed(10)

# 特殊标记：0，用于填充或者标记未知单词
# <SOS>: 句子起始标识符
# <EOS>：句子结束标识符
# <PAD>：补全字符
# <MASK>：掩盖字符
# <SEP>：两个句子之间的分隔符
# <UNK>：低频或未出现在词表中的词
vocab_size = len(vocab2int) + 1  # 词汇表大小
embedding_dim = 2  # 嵌入维度


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))  # 初始化embedding矩阵
        self.linear = nn.Linear(embedding_dim, vocab_size)  # vocab_size = 14

    def forward(self, context):
        # context.shape    ->   [bs, 2]
        context_emb = self.embeddings[context]  # 上下文的嵌入向量
        # print(context_emb.shape)  # torch.Size([4, 2, 2])
        avg_emb = torch.mean(context_emb, dim=1, keepdim=True).squeeze(1)
        y = self.linear(avg_emb)
        return y


model = CBOWModel(vocab_size, embedding_dim)
cri = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

bs = 4
epochs = 2000
for epoch in range(1, epochs + 1):
    total_loss = 0
    for batch_index in range(0, len(context), bs):
        # 上下文的Tensor
        context_tensor = torch.tensor(context[batch_index: batch_index + bs])
        # 中心词的tensor
        center_tensor = torch.tensor(center[batch_index: batch_index + bs])

        output = model(context_tensor)
        loss = cri(output, center_tensor)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(context)
    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}]  Loss {avg_loss:.4f}")
        # 每个单词的embedding向量
        word_vec = model.embeddings.data.numpy()
        # print(word_vec)
        x = word_vec[:, 0]
        y = word_vec[:, 1]
        selected_word = ["dog", "cat", "milk"]
        selected_word_index = [vocab2int[word] for word in selected_word]
        selected_word_x = x[selected_word_index]
        selected_word_y = y[selected_word_index]
        plt.cla()
        plt.scatter(selected_word_x, selected_word_y, color="blue")
        # 将每个点的标注加上
        for word, x, y in zip(selected_word, selected_word_x, selected_word_y):
            plt.annotate(word, (x, y), textcoords="offset points", xytext=(0, 10))
        plt.pause(0.5)

plt.show()