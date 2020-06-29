import torch

# [start:end:steps]
a = torch.rand(4, 3, 28, 28)
print(a.shape)
print(a[0, 0].shape)
print(a[:2, 1:, -5:, :].shape)
print(a[:, ::, 0:28:2, ::2].shape)

# index_select
# 选择哪个维度，然后在这个维度上面选择哪些项的索引（索引需要先转化成tensor类型）
b = torch.tensor([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]],
    [[13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24]]
])
print(b.shape)
b = b.index_select(0, torch.tensor([0, 2]))
print(b)
print(b.shape)

# ... ==> 任意

# masked_select ==> 打平
x = torch.randn(3, 4)
mask = x.ge(0.5)
print(mask)
x = torch.masked_select(x, mask)
print(x, x.shape)

# take ==> 先打平再按索引取
src = torch.tensor([[1, 2, 3], [4, 5, 6]])
get = torch.take(src, torch.tensor([0, 2, 5]))
print(get)


