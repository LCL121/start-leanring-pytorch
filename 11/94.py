from torch import nn
import torch

lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
print(lstm)
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)
# out: [10, 3, 20]
# h, c: [4, 3, 20]
print(out.shape, h.shape, c.shape)


print('one layer lstm')
x = torch.randn(10, 3, 100)
cell = nn.LSTMCell(input_size=100, hidden_size=20)
h = torch.zeros(3, 20)
c = torch.zeros(3, 20)
for xt in x:
    h, c = cell(xt, [h, c])
# h, c: [1, 3, 20]
print(h.shape, c.shape)


print('two layer lstm')
x = torch.randn(10, 3, 100)
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(h1, [h2, c2])
# h, c: [1, 3, 20]
print(h.shape, c.shape)

