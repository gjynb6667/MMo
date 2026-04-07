# 导入库
import os
import random

import matplotlib.pyplot as plt
import numpy as np

# 导入torch的相关库
import torch
from torchvision import datasets, transforms

from lenet5 import LeNet5

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns


# 设置随机数种子，保证结果的可复现
def setup_seed(seed):
    np.random.seed(seed)  # Numpy的随机数种子
    random.seed(seed)  # Python的随机数种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 配置Python的哈希种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置cuda随机数种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭cuddn加速
        torch.backends.cudnn.deterministic = True  # 设置cudnn为确定性算法


# 设置随机数种子
setup_seed(0)

# 检查GPU、CUDA是否可用，如果可用就是用GPU，否则使用CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available, Using CPU.")

# 数据读取
train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# # 显示6张图片
# examples = enumerate(test_loader)
# batch_idx, (imgs, labels) = next(examples)
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(imgs[i][0], cmap="gray", interpolation="none")
#     plt.title(f"Truth: {labels[i]}")
# plt.show()

model = LeNet5().to(device)
cri = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# epoches = 10
# for epoch in range(epoches):
#     model.train()
#     total_loss = 0
#     for i, (images, labels) in enumerate(train_loader):
#         # 将数据移动到设备上CPU/GPU
#         images = images.to(device)
#         labels = labels.to(device)
#         # 前向传播
#         outputs = model(images)
#         loss = cri(outputs, labels)
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # epoch的总损失
#         total_loss += loss
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch [{epoch + 1}/{epoches}], Loss: {avg_loss:.4f}")
#
# torch.save(model.state_dict(), "model.pth")
total = 0
corret = 0
predicted_labels = []
true_labels = []

model.load_state_dict(torch.load("model.pth"))
model.eval()

# 临时禁用自动梯度计算
# 1、节省内存
# 2、加速计算
# 3、防止梯度累加
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print(outputs.shape)  # [64, 10], [bs, labels]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        corret += (predicted == labels).sum().item()

        # 将数据扔回到CPU中，为了后续结果分析
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(f"Accuracy of the model on the test images {100 * corret / total} %")
conf_matrix = confusion_matrix(true_labels,predicted_labels)
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='b')
plt.xlabel("x")
plt.ylabel("y")
plt.title("conf_matrix")
plt.show()