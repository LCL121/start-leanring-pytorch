import torch

# add/minus/multiply/divide ==> 相同位置元素项运算
# + 或 add 会执行broadcasting
a = torch.ones(3, 4)
b = torch.zeros(4)
print((a + b).shape)
print(torch.add(a, b).shape)
print(torch.all(torch.eq(torch.add(a, b), (a + b))))
# - 或 sub 会执行broadcasting
print((b - a).shape)
print(torch.sub(a, b).shape)
print(torch.all(torch.eq(torch.sub(a, b), (a - b))))
# * 或 mul 会执行broadcasting
print((a * b).shape)
print(torch.mul(a, b).shape)
print(torch.all(torch.eq(torch.mul(a, b), (a * b))))
# / 或 div 会执行broadcasting
# / ==> 除法      // ==> 整除
print((b / a).shape)
print(torch.div(b, a).shape)
print(torch.all(torch.eq(torch.div(b, a), (b / a))))

# matmul  ==> 矩阵相乘
# mm ==> only for 2d （不推荐）
# 推荐matmul，@是matmul的重载运算符
c = torch.rand(4, 784)
d = torch.rand(512, 784)
print((c @ d.t()).shape)
e = torch.rand(4, 3, 28, 64)
f = torch.rand(4, 3, 64, 32)
print((e @ f).shape)
# @ 或 matmul 会执行broadcasting
g = torch.rand(64, 32)
print((e @ g).shape)

# pow ==> 每个元素项运算
# ** 或 pow
h = torch.full([2, 2], 3, dtype=torch.float)
print(h.pow(3))
print(h ** 4)

# sqrt/rsqrt ==> 每个元素项运算
# sqrt ==> 平方根
# rsqrt ==> 平方根的倒数
i = torch.full([2, 2], 2, dtype=torch.float)
ii = i ** 2
print(ii.sqrt())
print(ii.rsqrt())

# exp/log
# exp ==> e的多少次方，每一项当成一个指数
# log ==> 每一项进行loge运算
# log2
# log10
j = torch.tensor([1., 2., 3.])
print(torch.exp(j))
print(torch.log(torch.exp(j)))

# floor/ceil/round/trunc/frac
# floor ==> 向下取
# ceil ==> 向上取
# round ==> 四舍五入
# trunc ==> 取整数部分
# frac ==> 取小数部分
k = torch.tensor(3.14)
print(torch.floor(k), torch.ceil(k), k.round(), k.trunc(), k.frac())

# clamp ==> Clamp all elements in input into the range [ min, max ] and return a resulting tensor
# If input is of type FloatTensor or DoubleTensor, args min and max must be real numbers,
# otherwise they should be integers.
l = torch.rand(2, 3) * 15
print(l.max(), l.median(), l.clamp(10), l.clamp(0, 10))


