import torch

# broadcasting ===> unsqueeze && expand && without copying data
# if either has no dimesion, insert one dimension ahead
# expand dims with size 1 to same size
# [32, 1, 1] => [1, 32, 1, 1] => [4, 32, 14, 14]
# 低维度必须给，而且必须正确
# ①实际的扩展。②节省内存资源。当没有维度的时候，首先添加一个size=1的维度，然后对size=1的所有维度进行扩展。
# expand_as
a = torch.zeros(2, 1, 3)
print(a.shape)
b = torch.ones(3, 1)
print(b.shape)
c = torch.broadcast_tensors(a, b)
# broadcase_tensors ==> 返回 tuple（元组）==> 元组里面两个 tensor，分别对应两个输入
print(c[0], c[1])
print(c[0].shape, c[1].shape)
