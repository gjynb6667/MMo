import torch

tensor1 = torch.tensor([[1.1,2.2],[3.3,4.4]])
print(tensor1.shape)
print(tensor1.dtype)
a = torch.arange(20).reshape(4,5)
print(a.storage().data_ptr())
b = a.transpose(0,1)
print(b.storage().data_ptr())
#torch的不连续性
print(b)
print(b.flatten())
print(b.storage().tolist())
