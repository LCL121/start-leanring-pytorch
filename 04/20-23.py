import torch

# 1
# view reshape ===> 功能一样，只是不同版本的产物，保证numel()不变，即大小不变
# view ===> lost dim information 数据的存储/维度非常重要，需要时刻记住
a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.view(4, 1*28*28).shape)
print(a.view(4*1*28, 28).shape)
print(a.view(4*1, 28, 28).shape)


# 2
# squeeze unsqueeze
# unsqueeze(pos/index) 插入维度 ===> pos/index 为正数，则在position之前插入。为负数，则在position之后插入
#                                   范围 [-a.dim()-1, a.dim()+1)
# squeeze(pos/index) 维度删减 ===> pos/index 为空，则默认会删除所有可以删减的
print(a.unsqueeze(0).shape)
b = torch.rand(32)
print(b.shape)
b = b.unsqueeze(0).unsqueeze(2).unsqueeze(2)
print(b.shape)
print(b.squeeze().shape)
print(b.squeeze(0).shape)


# 3
# expand repeat
# expand ===> 只是改变了理解方式，没有改变数据，不会主动拷贝数据
#             扩展（expand）张量不会分配新的内存，只是在存在的张量上创建一个新的视图（view）
#             前后的dimension必须一致，1 ==> N
#             Passing -1 as the size for a dimension means not changing the size of that dimension.
#             Any dimension of size 1 can be expanded to an arbitrary value without allocating new memory.
# repeat ===> 拷贝，改变数据
#             参数是要拷贝的次数
#             Repeats this tensor along the specified dimensions.
c = torch.rand(1, 2, 1, 1)
print(c)
print(c.expand(1, 2, 1, 3).numel())
print(c.expand(-1, 2, 2, -1).shape)
print(c.repeat(1, 1, 1, 3).numel())
# ??? 为什么 expand 和 repeat 的 numel 一样


# 4
# t ==> 转置，只能用于 dimension = 2
d = torch.rand(2, 3)
print(d.shape)
print(d.t().shape)


# 5
# transpose  ==> 矩阵维度交换
e = torch.rand(1, 2, 3, 4)
print(e)
print(e.transpose(0, 3).shape)
print(e.transpose(0, 3).view(4, 2*3*1))
f = e.transpose(0, 3).view(4, 2*3*1).view(4, 2, 3, 1)
print(f)
print(torch.all(torch.eq(e.transpose(0, 3), f)))
print(torch.all(torch.eq(e, f.transpose(0, 3))))
# 这个地方好像和视频里面不太一样 ？？？？
# 不用 contiguous 也可以 ？？？？
g = e.transpose(0, 3).contiguous().view(4, 2*3*1).view(4, 2, 3, 1)
print(torch.all(torch.eq(g, f)))
print(torch.all(torch.eq(e, g.transpose(0, 3))))
# try this problem
# contiguous  ===> Returns a contiguous in memory tensor containing the same data as self tensor.
# Otherwise, contiguous() needs to be called before the tensor can be viewed.
# torch.view等方法操作需要连续的Tensor。有些tensor并不是占用一整块内存，而是由不同的数据块组成。
# 需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
print(e.shape)
print(e.contiguous().shape)


# 6
# permute ===> 直接给出矩阵维度变换的所有索引
# 如果需要多次transpose，可以用permute替换
h = torch.rand(2, 3, 4, 5, 6)
print(h.shape)
print(h.permute(4, 3, 2, 1, 0).shape)



