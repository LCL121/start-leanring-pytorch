import torch

# torch.where(condition, x, y)
cond = torch.rand(4, 4)
a = torch.zeros(4, 4)
b = torch.ones(4, 4)
print(torch.where(cond > 0.5, a, b))

# torch.gather(input, dim, index, out=None, sparse_grad=False)
# 高维度要对应，可以使得index和input可以映射到
label = torch.tensor([10, 30, 50, 70, 90, 110, 130])
index = torch.tensor([[0, 3, 3, 2, 1, 3, 2, 1, 3, 2, 1],
                      [2, 1, 3, 3, 2, 2, 1, 3, 2, 1, 3],
                      [4, 3, 2, 2, 1, 0, 3, 3, 2, 1, 3]])
result = torch.gather(input=label.expand(3, 7), dim=1, index=index)
print(result)
