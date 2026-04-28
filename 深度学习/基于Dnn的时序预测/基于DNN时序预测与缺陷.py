import  torch
import  numpy as  np
from torch import  nn
text =  "hey how are you"
input_seq = []
output_seq = []
window = 5
for i in range(0,len(text)-window,1):
    input_seq.append(text[i:i+window])
    output_seq.append(text[i+window])
chars = set(text)
chars = sorted(chars)
char2int = {char:ind for ind,char in enumerate(chars)}
print(char2int)
int2char = dict(enumerate(chars))
input_seq = [[char2int[char]for char in seq] for seq in input_seq]
print(input_seq)
output_seq = [[char2int[char]for char in seq] for seq in output_seq]
features = np.zeros((len(input_seq),len(chars)),dtype= np.float32)
for i,seq in enumerate(input_seq):
    features[i,seq] = 1.0
input_seq = torch.tensor(features,dtype=torch.float32)
features = np.zeros((len(output_seq),len(chars)),dtype= np.float32)
for i,seq in enumerate(output_seq):
    features[i,seq] = 1.0
output_seq = torch.tensor(features,dtype=torch.float32)
class DNN(nn.Module):
    def __init__(self,input_size ,hidden_size,output_size):
        super(DNN,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x =  nn.functional.relu(self.layer1(x))
        x =  self.layer2(x)
        return  x
model = DNN(len(chars),32,len(chars))
cri = nn.CrossEntropyLoss()
opitmizer =  torch.optim.Adam(model.parameters(),lr = 0.001)
epochs = 1000
for epoch in  range(1,epochs+1):
    output = model(input_seq)
    loss = cri(output,output_seq)
    opitmizer.zero_grad()
    loss.backward()
    opitmizer.step()
    if epoch%10 == 0 :
        print(f"epoch:{epoch}/{epochs},loss:{loss}")
input_text = "hey h"
input_text = [[char2int[char]for char in seq] for seq in input_text]
features = np.zeros(len(chars),dtype= np.float32)
for seq in input_text:
    features[seq] = 1.0
input_text = torch.tensor(features,dtype=torch.float32)
print(input_text)
out = model(input_text)
print(int2char[torch.argmax(out).item()])