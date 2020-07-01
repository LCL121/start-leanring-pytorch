import torch
import torch.nn.functional as F


def get_entropy(x):
    return -(x * torch.log2(x)).sum()


a = torch.full([4], 0.25)
b = get_entropy(a)
print(b)

a = torch.tensor([0.1, 0.1, 0.7, 0.1])
b = get_entropy(a)
print(b)

a = torch.tensor(([0.001, 0.001, 0.001, 0.999]))
b = get_entropy(a)
print(b)


x = torch.randn(1, 784)
w = torch.randn(10, 784)
logits = x @ w.t()
print(logits)
# F.cross_entropy ==> 已经把 softmax->log->nll_loss 打包在一起了
cross_entropy = F.cross_entropy(logits, torch.tensor([3]))
print(cross_entropy)
#
pred = F.softmax(logits, dim=1)
print(pred)
pred_log = torch.log(pred)
print(pred_log)
# nll_loss ==> NLLLoss: 最大似然 / log似然代价函数
nll_loss = F.nll_loss(pred_log, torch.tensor([3]))
print(nll_loss)

# CrossEntropyLoss: 交叉熵损失函数。
# 损失函数 CrossEntropyLoss() 与 NLLLoss()类似, 唯一的不同是它为我们去做 softmax并取对数.
# CrossEntropyLoss()=log_softmax() + NLLLoss()
