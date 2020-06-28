import numpy as np
import torch

# 创建Tensor

# 1. import from numpy  ===> use torch.from_numpy()
a = np.array([2, 3.3])
a = torch.from_numpy(a)
print(a)

# 2. import from list
# .tensor 参数是现成的数据: 标量、list、numpy
b = torch.tensor([2, 3.2])
# .Tensor 参数为list、numpy，则和.tensor一样，为标量则当成shape来生成数据
# 推荐小写用在现成的数据转化，大写用在shape

# 3. uninitialized
c = torch.empty(1, 4, 5)
d = torch.FloatTensor(1, 4, 5)
e = torch.IntTensor(1, 4, 5)
print(c)
print(d)
print(e)

# .Tensor 和 .tensor 默认是floatTensor，可以自己设置默认的
torch.set_default_tensor_type(torch.DoubleTensor)

# 4. rand/rand_like,randint,randn,full,arange/range,linspace/logspace,ones/zeros/eye，randperm
# rand 会随机产生0～1之间的数值，不包括1（均匀分布）
# rand_like 先将传入参数的shape读出来，再传入rand函数
# randint
# randn （正态分布）
# full 全是某个值
# arange,linspace （等差数列）
# logspace 以等差数列为计算的log值，base可以为2, 10, e
# ones 全是1
# zeros 全是0
# eye 对角线全是1
# randperm 用于打散
f = torch.rand(3, 4, 5)
g = torch.rand_like(f)
print(g.shape)
h = torch.randint(1, 10, [3, 3])
print(h)
i = torch.full((3, 4), 7, dtype=torch.int)
print(i)
j = torch.arange(0, 10, 2)
print(j)
k = torch.linspace(0, 10, steps=3)
print(k)
l = torch.logspace(0, 10, steps=2, base=10)
print(l)

try1 = torch.tensor([1, 2, 3])
try2 = torch.tensor([10, 20, 30])
idx = torch.randperm(3)
print(idx)
print(try1, try2)
try1 = try1[idx]
try2 = try2[idx]
print(try1, try2)







