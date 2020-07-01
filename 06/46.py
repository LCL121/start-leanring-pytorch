from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from matplotlib import pyplot as plt


def himmelbau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


# 从-6到6，所有相差0.1的值，[-6, 6)
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
print(X, torch.tensor(X).shape, end="\n\n")
print(Y, torch.tensor(Y).shape)
# 矩阵的每一项的加减乘除运算
Z = himmelbau(X, Y)
print(Z)

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


x = torch.tensor([0., 0.], requires_grad=True)
# 使用一个优化器，自动完成 x' = x - 0.001 * x的导 和 y' = y - 0.001 * y的导
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
    pred = himmelbau(x[0], x[1])
    # 梯度清零
    optimizer.zero_grad()
    # 方向传播，生成x/y的梯度信息
    pred.backward()
    # 更新x'，y'
    optimizer.step()
    if step % 2000 == 0:
        # tolist 的作用： tensor ==> list or number
        # item 的作用： tensor ==> number  This only works for tensors with one element.For other cases, see tolist().
        print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))
