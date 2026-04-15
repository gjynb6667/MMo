import os
import random

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 导入深度学习框架 PyTorch 相关库
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from  ResNet网络结构 import  ResNet,BasicBlock
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True
set_seed(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is available. Using GPU.")
transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ]),
    "val":transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
}
train_dir = './dataset/dataset/train'
val_dir = './dataset/dataset/val'
train_dataset = datasets.ImageFolder(
    train_dir,
    transform = transforms["train"]
)
val_dataset = datasets.ImageFolder(
    val_dir,
    transform = transforms["val"]
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
exanple = enumerate(train_loader)
batch_idx,(image,label) = next(exanple)
fig = plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(image[i][0],cmap = 'gray')
    plt.title(f"Ground Truth:{label[i]}")
    plt.xticks([])
    plt.yticks([])
plt.show()
model = ResNet(BasicBlock,[2,2,2,2],num_classes= 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
save_path = './model'
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i,(images,labels) in enumerate(train_loader) :
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f'Epoch[{epoch+1}/{epochs}] Batch [{i+1}/{len(train_loader)}] loss{loss.item():.4f}')
    avg_loss = total_loss/len(train_loader)
torch.save(model.state_dict(),save_path)
model.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for images,labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
print(f"Accuracy of the model no test images:{100*correct/total:.2f}")

conf = confusion_matrix(true_labels, predicted_labels)
# 可视化
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",cbar = False)
plt.xlabel("predict")
plt.ylabel("true")
plt.show()