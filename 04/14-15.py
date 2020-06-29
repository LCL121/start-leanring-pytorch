import torch
import numpy as np

# 2行3列 随机的正态分布
a = torch.rand(2, 3)
print(a)
print(a.type())
print(type(a))
# 变量合法化检验
print(isinstance(a, torch.FloatTensor))

# 一样的数据放的位置不一样，导致类型不一样
# 默认是数据是CPU上的
print(isinstance(a, torch.cuda.FloatTensor))
a = a.cuda()
print(isinstance(a, torch.cuda.FloatTensor))

b = torch.tensor(1.3)
print(b.shape)  # torch.Size([])
print(len(b.shape))
print(b.dim())  # .dim() 显示维度大小
print(b.size())  # torch.Size([])

# .tensor 接受数据的内容
c = torch.tensor([1.1])
print(c, c.shape, c.dim())  # tensor([1.1000]) torch.Size([1]) 1
# .FloatTenor 接受的是数据的shape
d = torch.FloatTensor(1)
print(d, d.shape, c.dim())  # tensor([0.]) torch.Size([1]) 1
e = torch.FloatTensor(2)
print(e, e.shape, e.dim())  # tensor([4.4614e-16, 0.0000e+00]) torch.Size([2]) 1
f = np.ones(2)
ff = torch.from_numpy(f)
print(f, ff, ff.shape, ff.dim())  # [1. 1.] tensor([1., 1.], dtype=torch.float64) torch.Size([2]) 1

g = torch.IntTensor(2, 3)
print(g, g.size(), g.shape, g.dim())  # torch.Size([2, 3]) 2
print(g.size(0), g.size(1), g.shape[0], g.shape[1])

# 随机的均匀分布
h = torch.rand(2, 3, 4)
print(h, h.shape, h.dim())
print(type(h.shape), type(list(h.shape)))  # 使用list函数直接将 torch.Size类型 ===> list类型

# other
print(h.numel())  # Returns the total number of elements in the input tensor. 2 * 3 * 4 = 24



