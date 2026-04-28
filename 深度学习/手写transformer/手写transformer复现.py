import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  math
"""
1.把各个数据进行赋值
[bs,seq_len,dim]
[bs,trg_len,dim]
N,layer后的dim
src_data,trg_data
2.先把主transformer写了
调用解码器和编码器
forward的方法要导入seq_len,trg_len
先还要创建一个方法来创建mask
3.根据执行顺序先数据进入
而数据需要进行position的绝对位置编码和Enbedding向量化
而向量化可以在Encoder类内执行所以先写position方法
4.进入Encoder和Decoder方法
进入先要向量，position，然后,mulit
所以再写个mulit
需要输入一个三维的数据
有2个Linear层
先确定输入q（bs），k(seq_len),v(d_model)
经过Linear
再拆分（需要进行头拆分要用到，n_head）
5.free
6.创建encoderlayer
"""
d_model = 512
n_heads = 8
n_layer = 6
d_ff = 2048
bs = 32
seq_len = 20
trg_len = 10
input_value = 10000
out_value = 10000
src_data = torch.randint(0,input_value,(bs,seq_len))
trg_data = torch.randint(0,input_value,(bs,trg_len))
class Transformer(nn.Module):
    def __init__(self,d_model,n_layer,d_ff,n_heads,input_value,out_value):
        super(Transformer,self).__init__()
        self.encoder = Encoder(d_model,d_ff,input_value,n_layer,n_heads)
        self.Decoder = Decoder(d_model,n_heads,d_ff,n_layer,out_value)
    def forward(self,src,trg):
        src_mask = src.unsqueeze(1).unsqueeze(2)
        out = self.encoder(src,src_mask)
        target_mask = self.create_mask(trg)
        out_put = self.Decoder(trg,out,src_mask,target_mask)
        return out_put
    def create_mask(self,trg_data):
        trg_pad_mask = trg_data.unsqueeze(1).unsqueeze(2)
        trg_len = trg_data.size(1)
        trg_seq_mask = torch.tril(torch.ones(trg_len,trg_len)).bool()
        return trg_seq_mask
class Position(nn.Module):
    def __init__(self,d_model,number = 5000):
        super(Position,self).__init__()
        self.encoding  = torch.zeros(number,d_model)
        position = torch.arange(0,number).unsqueeze(1).float()
        div_dam = torch.arange(0,d_model,2).float()/d_model
        self.encoding[:,0::2] = torch.sin(position/torch.pow(10000.0,div_dam))
        self.encoding[:,1::2] = torch.cos(position/torch.pow(10000.0,div_dam))
        self.encoding = self.encoding.unsqueeze(0)
    def forward(self,x):
        x = x+ self.encoding[:,x.size(1)]
        return x
class Encoder(nn.Module):
    def __init__(self,d_model,d_ff,input_value,n_layer,n_heads):
        super(Encoder,self).__init__()
        self.enbedding = nn.Embedding(input_value,d_model)
        self.position = Position(d_model)
        self.transformer = nn.ModuleList([EncoderLayer(d_model,n_heads, d_ff)for  _ in range(n_layer)])

    def forward(self,x,mask =None):
        x = self.enbedding(x)
        x = self.position(x)
        for layer in self.transformer:
            x = layer(x,mask)
        return x
class Decoder(nn.Module):
    def __init__(self,d_model,n_head,d_ff,n_layer,out_value):
        super(Decoder,self).__init__()
        self.embedding = nn.Embedding(out_value,d_model)
        self.position = Position(d_model)
        self.transformer = nn.ModuleList([DecoderLayer(d_model,n_head, d_ff)for  _ in range(n_layer)])
        self.linear = nn.Linear(d_model,out_value)
    def forward(self,x,encoder_out,src_mask,trg_mask):
        x = self.embedding(x)
        x = self.position(x)
        for layer in self.transformer:
            x = layer(x,encoder_out,src_mask,trg_mask)
        x = self.linear(x)
        return x
class Mulit(nn.Module):
    def __init__(self,d_model,n_head):
        super(Mulit,self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.number = self.d_model//self.n_head
        self.WQ = nn.Linear(d_model,d_model)
        self.WK = nn.Linear(d_model,d_model)
        self.WV = nn.Linear(d_model,d_model)
        self.Linear = nn.Linear(d_model,d_model)
    def forward(self,q,k,v,mask = None):
        bs = q.size(0)
        Q = self.WQ(q)
        K = self.WQ(k)
        V = self.WQ(v)
        Q = Q.view(bs,-1,self.n_head,self.number).transpose(1,2)
        K = K.view(bs,-1,self.n_head,self.number).transpose(1,2)
        V = V.view(bs,-1,self.n_head,self.number).transpose(1,2)
        scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.n_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0 ,float("-1e20"))
        weights = F.softmax(scores,dim= -1)
        attention = torch.matmul(weights,V)
        attention = attention.transpose(1,2).contiguous().view(bs,-1,d_model)
        output = self.Linear(attention)
        return  output
class Free(nn.Module):
    def __init__(self,d_model,d_ff):
        super(Free,self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_head,d_ff):
        super(EncoderLayer,self).__init__()
        self.mult = Mulit(d_model,n_head)
        self.dropout = nn.Dropout(0.1)
        self.LN1 = nn.LayerNorm(d_model)
        self.free = Free(d_model,d_ff)
        self.LN2 = nn.LayerNorm(d_model)
    def forward(self,x,mask = None):
        attention = self.mult(x,x,x,mask)
        x = x + self.dropout(attention)
        x = self.LN1(x)
        x = x + self.free(x)
        out = self.LN2(x)
        return out
class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_head,d_ff):
        super(DecoderLayer,self).__init__()
        self.mult = Mulit(d_model,n_head)
        self.dropout = nn.Dropout(0.1)
        self.LN1 = nn.LayerNorm(d_model)
        self.Mult = Mulit(d_model,n_head)
        self.LN2 = nn.LayerNorm(d_model)
        self.free = Free(d_model,d_ff)
        self.LN3 = nn.LayerNorm(d_model)
    def forward(self,x,encoder_out,src_mask = None,trg_mask = None):
        attention = self.mult(x,x,x,mask = trg_mask)
        x = x + self.dropout(attention)
        x = self.LN1(x)
        x = self.Mult(x,encoder_out,encoder_out,mask = src_mask)
        x = self.LN2(x)
        x = x + self.free(x)
        out = self.LN3(x)
        return out
transformer = Transformer(d_model,n_layer,d_ff,n_heads,input_value,out_value)
out_put = transformer(src_data,trg_data)
print(out_put.shape)



