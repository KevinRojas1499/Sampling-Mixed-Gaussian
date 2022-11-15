import torch 
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[5,6],[7,8]])

c = torch.cat((a,b),dim=1)
print(c)

d,e = torch.chunk(c,2,dim=1)

print(d,e)