import torchvision
import torch
import torch.nn as nn

# mean 给定数据的均值， std 给定数据的方差
# image=(image-mean)/std
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


x = torch.rand(100, 16, 784)
layer = nn.BatchNorm1d(16)
out = layer(x)
print(layer.running_mean)
print(layer.running_var)

x = torch.rand(1, 16, 7, 7)
layer = nn.BatchNorm2d(16)
out = layer(x)
print(layer.weight)
print(layer.weight.shape)
print(layer.bias.shape)
print(vars(layer))


# 在test模式，要layer.eval() ==> 对BatchNorm进行变换
