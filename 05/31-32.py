import torch

# norm ==> Returns the matrix norm or vector norm of a given tensor.
# 默认是2-范数
# dime：可以指定维度上的范数
a = torch.full([8], 1, dtype=torch.float)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(a.shape, b.shape, c.shape)
print(a.norm(1), b.norm(1), c.norm(1))
print(a.norm(), b.norm(), c.norm())
print(b.norm(1, dim=1))
print(b.norm(2, dim=1))

# median ==> 所有项的中间值
# mean ==> 所有项的平均值
# prod ==> 所有项的累乘
d = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
print(d.median(), d.mean(), d.prod())

# max ==> 所有项的最大值
# min ==> 所有项的最小值
# argmax ==> 所有项的最大值的索引
# argmin ==> 所有项的最小值的索引
# 说明求最大值或最小值之前是先将维度打平，
# 可以加dim参数 ==> 哪个维度上的最大值，
# keepdim参数 ==> 保证dimension和原来的数据一样
print(d.argmax(), d.argmin())
print(d.argmax(dim=1), d.argmin(dim=1))

# topk ==> 取前面的几个值
# largest ==> 默认是True。True，则取大的；False，则取小的
e = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]], dtype=torch.float)
print(e.topk(3, dim=1))
print(e.topk(3, dim=1, largest=False))

# kthvalue ==> 取第几小的值
# 取输入张量input指定维度上第k个最小值。如果不指定dim。默认为最后一维。
# 返回一个元组(value, indices), 其中indices是原始输入张量中沿dim维的第k个最小值下标。
print(e.kthvalue(3, dim=1))

# >, >=, <, <=, !=, ==
# 每一项都和给定的标量进行比较
print(e > 5)

# eq ==> 两个tensor对应位置的每一项进行比较

