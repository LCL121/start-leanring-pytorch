import torch
from torch.nn import functional as F

# torch.sigmoid
# linspace ==> Returns a one-dimensional tensor of steps equally spaced points between start and end.
a = torch.linspace(-100, 100, 10)
print(a)
print(torch.sigmoid(a))
# UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
# print(F.sigmoid(a))

# torch.tanh
b = torch.linspace(-1, 1, 10)
print(torch.tanh(b))

# torch.relu
print(torch.relu(b))
print(F.relu(b))

# F.mse_loss
# 一、autograd.grad
#   必须告诉pytorch那些需要求导
#   1：初始化时，requires_grad=True
#   2：中间，requires_grad_()，但是需要把基于它的网络（求导的网络）重新计算
x = torch.ones(1)
# w = torch.full([1], 2, dtype=torch.float, requires_grad=True)
w = torch.full([1], 2, dtype=torch.float)
mse = F.mse_loss(torch.ones(1), x * w)
print(mse)
w.requires_grad_()
mse = F.mse_loss(torch.ones(1), x * w)
print(torch.autograd.grad(mse, [w]))
# 二、loss.backward
x = torch.ones(1)
w = torch.full([1], 2, dtype=torch.float)
mse = F.mse_loss(torch.ones(1), x * w)
w.requires_grad_()
mse = F.mse_loss(torch.ones(1), x * w)
print(mse)
mse.backward()
print(w.grad)

# F.softmax
c = torch.rand(3, requires_grad=True)
p = F.softmax(c, dim=0)
print(p)
# 如果多次使用backward，所以要retain_graph=True，保存requires_grad标志
# backward 和 autograd.grad 输入必须是标量
p[1].backward(retain_graph=True)
print(c.grad)
p[0].backward()


