import  os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as  np
from torch.utils.data import  Dataset,Subset,DataLoader
import pandas as  ps
from  sklearn.preprocessing import  MinMaxScaler
import  matplotlib.pyplot as plt
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
class PowerData(Dataset):
    def __init__(self,csv_path,input_len):
        self.input_len = input_len
        self.data = pd.read_csv(csv_path)
        self.data["功率(kW)"] = np.minimum(self.data["功率(kW)"],1500)
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.data["power_normalized"] = self.scaler.fit_transform(self.data["功率(kW)"].values.reshape(-1,1))
    def __len__(self):
        return  len(self.data) - self.input_len
    def __getitem__(self,idx):
        start_idx = idx
        end_idx = idx+ self.input_len
        feature = self.data["power_normalized"].values[start_idx:end_idx]
        target =  self.data["power_normalized"].values[end_idx:end_idx+1]
        return torch.tensor(feature,dtype=torch.float32),torch.tensor(target,dtype=torch.float32)

int_put = 6
power_data = PowerData("./A01.csv",int_put)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
tarin_size = int(len(power_data)*train_ratio)
val_size = int(len(power_data)*val_ratio)
test_size = int(len(power_data)*test_ratio)
indices = list(range(len(power_data)))
train_dataset = Subset(power_data,indices[:tarin_size])
val_dataset = Subset(power_data,indices[tarin_size:tarin_size+val_size])
test_dataset = Subset(power_data,indices[tarin_size+val_size:])
train_dataloder = DataLoader(train_dataset,batch_size=64,shuffle=True)
val_dataloder = DataLoader(val_dataset,batch_size=64,shuffle=False)
test_dataloder = DataLoader(test_dataset,batch_size=1,shuffle=False)
class DNN(nn.Module):
    def __init__(self,input_size = 6,hidden_size =128,output_size =1):
        super(DNN,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x =  F.relu(self.layer1(x))
        x =  self.layer2(x)
        return  x
model = DNN().to(device)
crition = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,weight_decay=1e-5)
epochs = 50
for epoch in range(1,epochs+1):
    model.train()
    total_loss = 0
    for batch_feature,batch_target in train_dataloder:
        batch_feature,batch_target = batch_feature.to(device),batch_target.to(device)
        y_hat = model(batch_feature)
        loss = crition(y_hat.squeeze(1),batch_target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss/len(train_dataloder)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_feature, batch_target in val_dataloder:
            batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
            y_hat = model(batch_feature)
            loss = crition(y_hat.squeeze(1), batch_target.view(-1))
            total_loss += loss.item()
    val_loss = total_loss/len(val_dataloder)
    print(f'Epoch:{epoch}/{epochs},train_loss:{train_loss},val_loss:{val_loss}')
model.eval()
predict_list = []
target_list = []
with torch.no_grad():
    for batch_feature, batch_target in test_dataloder:

        batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
        y_hat = model(batch_feature)
        predict_list.append(y_hat.squeeze(1).item())
        target_list.append(batch_target.item())
predict_list = power_data.scaler.inverse_transform(np.array(predict_list).reshape(-1,1))
target_list = power_data.scaler.inverse_transform(np.array(target_list).reshape(-1,1))
plt.plot(target_list,label = "True")
plt.plot(predict_list,label = "predict")
plt.xlabel("Time")
plt.ylabel("Power")
plt.legend()
plt.show()



