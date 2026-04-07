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
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    "val":transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}
train_dir = './dataset/flower_data/train'
val_dir = './dataset/flower_data/val'
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms = transforms["train"]
)
val_dataset = datasets.ImageFolder(
    val_dir,
    transforms = transforms["val"]
)
displayed_labels = []
count = 0
fig,axes = plt.subplots(1,4,figsize=(15,3))
while count<4:
    index = np.random.randint(len(train_dataset))
    image,label = train_dataset[index]
    print(label)
    if label not in displayed_labels:
        mean = np.array([0.5,0.5,0.5])
        std = np.array([0.5,0.5,0.5])
        image = image.numpy()*std[:,None,None]+mean[:,None,None]
        image = np.transpose(image,(1,2,0))
        axes[count].imshow(image)
        axes[count].set_title(train_dataset.classes[label])
        axes[count].axis('off')
        displayed_labels.append(label)
        count +=1
plt.show()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
class AlexNet(nn.Module):
    def __init__(self,num_classes = 1000):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,96,11,4,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),
        )
        def forward(self,x):
            x = self.features(x)
            x = torch.flatten(x,1)
            x = self.classifier(x)
            return x
save_path = './model'
if not os.path.exists(save_path):
    os.makedirs(save_path)
model = AlexNet(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr =0.001 )
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(train_loader)
    if(epoch+1)%1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
torch.save(model.state_dict(),save_path+'/AlexNet.pth')
correct = 0
total = 0
predicted_labels = []
true_labels = []
model.eval()
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
        print(f'Accuracy of the model on the test images: {100 * correct / total:.4f} %')
