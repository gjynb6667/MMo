import  torch
import  numpy as  np
from torch import  nn
text =  "hey how are you"
input_seq = [text[:-1]]
output_seq = [text[1:]]
chars = set(text)
chars = sorted(chars)
char2int = {char:ind for ind,char in enumerate(chars)}
print(char2int)
int2char = dict(enumerate(chars))
input_seq = [[char2int[char]for char in seq] for seq in input_seq]
print(input_seq)
output_seq = [[char2int[char]for char in seq] for seq in output_seq]
def one_hot_encode(seq,bs,seq_len,input_size):
    x = np.zeros((bs,seq_len,input_size),dtype=np.float32)
    for i in range(bs):
        for u in range(seq_len):
            x[i,u,seq[i][u]] = 1.0
    return torch.tensor(x,dtype=torch.float32)
input_seq = one_hot_encode(input_seq,1,len(text)-1,len(chars))
output_seq = torch.tensor(output_seq,dtype=torch.long).view(-1)
class RNN(nn.Module):
    def __init__(self,input_size ,hidden_size,output_size):
        super(RNN,self).__init__()
        self.rnn1 = nn.RNN(input_size,hidden_size,num_layers=1,batch_first=True)
        self.hidden = hidden_size
        self.layer2 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x,hidden =  self.rnn1(x)
        x = x.contiguous().view(-1,self.hidden)
        x = self.layer2(x)
        return  x
model = RNN(len(chars),32,len(chars))
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
input_text = "hey how a"
to_be_pre_len = 2
for i in range(to_be_pre_len):
    chars = [char for char in input_text]
    character = np.array([[char2int[c] for c in chars]])
    character = one_hot_encode(character,1,character.shape[1],9)
    out = model(character)
    char_index = torch.argmax(out[-1]).item()
    input_text += int2char[char_index]
print(input_text)
