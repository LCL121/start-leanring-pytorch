import torch
from torch import nn

# 100: input_size ==> feature size
# 10: hidden_size ==> memory/hidden size
rnn = nn.RNN(100, 10)
print(rnn._parameters.keys())
print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)
print(rnn.bias_hh_l0.shape, rnn.bias_ih_l0.shape)


# 单层RNN
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
# 10个单词，3个句子，100个特征
x = torch.randn(10, 3, 100)
# 1层，3句子，20memory
out, h = rnn(x, torch.zeros(1, 3, 20))
print(out.shape, h.shape)


# 4层RNN
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
# 10个单词，3个句子，100个特征
x = torch.randn(10, 3, 100)
# 1层，3句子，20memory
out, h = rnn(x, torch.zeros(4, 3, 20))
# out ==> 最后一层的所有状态的输出（所有时间点的最后的状态）
# h ==> 最后一个时间点的所有memory的状态
print(out.shape, h.shape)


#
x = torch.randn(10, 3, 100)
cell1 = nn.RNNCell(100, 20)
h1 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
print(h1.shape)

#
x = torch.randn(10, 3, 100)
# 100 特征 ==> 30 memory ==> 20 memory
cell1 = nn.RNNCell(100, 30)
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
    h2 = cell2(h1, h2)
print(h1.shape, h2.shape)

