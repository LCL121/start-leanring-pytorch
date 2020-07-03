import torch
import torch.nn as nn

a = torch.rand(3, 2, 28, 28)
print(a.view(3, 2, 28, -1).shape)
print(a.view(3, 2, -1).shape)
print(a.view(3, -1).shape)
print(a.view(-1).shape)

print(a.view(3, -1, 28).shape)
print(a.view(-1, 28, 28).shape)

# view 中有参数 -1 ==> 打平操作


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


# 需要调用 nn.Paramter()，把tensor加到Module的paramters里面（并且会自动加上requires_grad=True），这样子数据才可以被优化器优化
class MyLinear(nn.Module):
    def __init__(self, input, output):
        super(MyLinear, self).__init__()
        self.w = nn.Parameter(torch.randn(output, input))
        self.b = nn.Parameter(torch.randn(output))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


myLinear = MyLinear(3, 4)
print(myLinear.w.shape)
print(myLinear.b.shape)
out = myLinear.forward(torch.randn(4, 3))
print(out.shape)
