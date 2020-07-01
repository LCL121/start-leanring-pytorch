import torch
import torch.nn.functional as F

# 单层感知机
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)
o = torch.sigmoid(x @ w.t())
# 1 与 o 比
loss = F.mse_loss(torch.ones(1, 1), o)
loss.backward()
print(w.grad)

# 我觉得应该是10个输入，2个神经元，2个输出结果
# 每个输入有2个权值，分别对应两个神经元
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
o = torch.sigmoid(x @ w.t())
print(o)
loss = F.mse_loss(torch.ones(1, 2), o)
loss.backward()
print(w.grad)

