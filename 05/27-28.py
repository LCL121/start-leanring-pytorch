import torch

# cat ==> 第一个参数：一个tensor的list
# 合并的维度可以不一样
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
c = torch.cat([a, b], dim=0)
print(c.shape)

# stack
# create new dimension
# Concatenates sequence of tensors along a new dimension.
# All tensors need to be of the same size.
a = torch.cat([a, torch.rand(1, 32, 8)])
d = torch.stack([a, b], dim=0)
print(d.shape)

# split ==> 按长度拆分
# 拆分的长度一样，参数为标量即可
# 拆分的长度不一样，参数也是一个list
# example 1
e = d.split(1, dim=0)
print(e[0].shape, e[1].shape)
f = torch.cat([d, e[0]])
print(f.shape)
# example 2
g1, g2 = f.split([1, 2], dim=0)
print(g1.shape, g2.shape)

# chunk ==> 按数量拆分
h1, h2 = d.chunk(2, dim=0)
print(h1.shape, h2.shape)
g1, g2, g3 = f.chunk(3, dim=0)
print(g1.shape, g2.shape, g3.shape)
