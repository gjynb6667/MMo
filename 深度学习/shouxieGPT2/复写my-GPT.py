import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
"""
0.导入相对应的参数
1.先写主体函数GPT2
内容是调用函数方法
显示进行向量化和位置编码
再进行layer——normalize
然后attention
再进行layer——normalize
mlp
layer
"""
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
class GPT(nn.Module):
    def __init__(self,config):
        super(GPT,self).__init__()
        self.config = config
        #ModuleList(dict())把字典打包为pytorch能识别的神经网络模块
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.vocab_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.ln2 = nn.Linear(config.n_embd,config.vocab_size,bis = False)
    def forward(self,idx):
        B,T = idx.size()
        assert T <= self.config.block_size, f"不能让seq_len {T} 大于 block_size {self.config.block_size}"
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        idx_emb = self.transformer.wte(idx)
        x = pos_emb+idx_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.ln2(x)
        return x
class Block(nn.Module):
    def __init__(self,config):
        super(Block,self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super(CausalSelfAttention,self).__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    def forward(self,x):
        B,T,C  = x.size()
        qkv = self.c_attn(x)
        Q,K,V = qkv.split(self.n_embd,dim = 2)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (Q@ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ V  # (B, nh, T, T) X (B, nh, T, hs) -> (B,  nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
class MLP(nn.Module):
    def __init__(self,config):
        super(MLP,self).__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("从预训练的GPT中加载模型:", model_type)

        # 根据模型类型确认参数
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M param
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M param
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M param
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M param
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        # 创建GPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # 从huggingface/transformers模型中初始化
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 将参数逐一对齐并复制
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"键不匹配, {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # openai使用了一个叫conv1d的模型，功能与linear一致，我们使用linear，需要单独处理它。需要转置
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:  # 其余的直接复制
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


model = GPT.from_pretrained("gpt2")
print("没报错！！！")
model.eval()
model.to("cpu")
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
print(tokens)
tokens = tokens.unsqueeze(0)
x = tokens.to("cpu")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
max_length = 30
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

tokens = x[0, :max_length].tolist()
decoded = enc.decode(tokens)
print(">", decoded)

