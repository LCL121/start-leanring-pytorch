import torch
import torch.nn as nn


x = torch.rand(1, 1, 28, 28)
layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
out = layer.forward(x)
print(out.shape)
layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)
print(out.shape)
layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)
print(out.shape)
# 推荐直接使用 layer()
# layer ==> hooks ==> forward
out = layer(x)
print(out.shape)
print(layer.weight.shape)
print(layer.bias.shape)

