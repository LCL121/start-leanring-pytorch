import torch
from torch import nn
import torch.nn.functional as F

x = torch.rand(1, 784)
layer1 = nn.Linear(784, 200)
layer2 = nn.Linear(200, 200)
layer3 = nn.Linear(200, 10)
print(x.shape)
x = layer1(x)
# inplace – can optionally do the operation in-place. Default: False
x = F.relu(x, inplace=True)
print(x.shape)
x = layer2(x)
x = F.relu(x, inplace=True)
print(x.shape)
x = layer3(x)
x = F.relu(x, inplace=True)
print(x.shape)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x
