# 导入所需的库
import os
import random

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np

# 导入深度学习框架 PyTorch 相关库
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 设置随机种子以保证结果的可重复性
def setup_seed(seed):
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True  # 设置 cudnn 为确定性算法


# 设置随机种子
setup_seed(0)
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}

train_dataset = datasets.ImageFolder("./dataset/train", transform=transform["train"])
test_dataset = datasets.ImageFolder("./dataset/test", transform=transform["test"])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印一下图片
examples = enumerate(test_dataloader)
batch_idx, (imgs, labels) = next(examples)
for i in range(4):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = imgs[i].numpy() * std[:, None, None] + mean[:, None, None]
    # 将图片转成numpy数组，主要是转换通道和宽高位置
    image = np.transpose(image, (1, 2, 0))
    plt.subplot(2, 2, i + 1)
    plt.imshow(image)
    plt.title(f"Truth: {labels[i]}")
plt.show()